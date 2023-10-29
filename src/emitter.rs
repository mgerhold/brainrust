use inkwell::context::Context;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use inkwell::module::Linkage;
use thiserror::Error;

use crate::command_line_arguments::{CommandLineArguments, EmitTarget};
use crate::emitter::state::{Function, State};
use crate::program::Program;

#[derive(Error, Debug)]
pub(crate) enum EmitError {
    FailedToWriteToFile {
        filename: PathBuf,
        error_message: String,
    },
    ModuleVerificationFailed(String),
}

impl Display for EmitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::FailedToWriteToFile {
                filename,
                error_message,
            } => {
                write!(
                    f,
                    "failed to write contents to file '{}': {}",
                    filename.display(),
                    error_message
                )
            }
            EmitError::ModuleVerificationFailed(error) => {
                write!(f, "module verification failed: {error}")
            }
        }
    }
}

mod state {
    use crate::emitter::EmitError;
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::module::Module;
    use inkwell::passes::{PassManager, PassManagerBuilder};
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };
    use inkwell::types::{BasicType, IntType, PointerType, VoidType};
    use inkwell::values::FunctionValue;
    use inkwell::{AddressSpace, OptimizationLevel};
    use std::path::Path;

    pub(super) enum DataType {
        Void,
        Char,
        Int,
        Size,
        Pointer,
    }

    pub(super) struct Function<'a> {
        context: &'a Context,
        builder: &'a Builder<'a>,
        function_value: FunctionValue<'a>,
    }

    impl<'a> Function<'a> {
        pub(super) fn new(
            context: &'a Context,
            builder: &'a Builder<'a>,
            function_value: FunctionValue<'a>,
        ) -> Self {
            Self {
                context,
                builder,
                function_value,
            }
        }

        pub(super) fn append_basic_block(&self, name: &str) -> &'a Builder<'a> {
            let block = self.context.append_basic_block(self.function_value, name);
            self.builder.position_at_end(block);
            self.builder
        }
    }

    impl<'a> From<Function<'a>> for FunctionValue<'a> {
        fn from(value: Function<'a>) -> Self {
            value.function_value
        }
    }

    pub(super) struct State<'a> {
        pub(super) context: &'a Context,
        pub(super) builder: Builder<'a>,
        pub(super) module: Module<'a>,
        target_machine: TargetMachine,
        pub(super) void_type: VoidType<'a>,
        pub(super) char_type: IntType<'a>,
        pub(super) int_type: IntType<'a>,
        pub(super) size_type: IntType<'a>,
        pub(super) pointer_type: PointerType<'a>,
    }

    impl<'a> State<'a> {
        pub(super) fn new(context: &'a Context, module_name: &str) -> Self {
            let builder = context.create_builder();
            let module = context.create_module(module_name);

            let default_triplet = TargetMachine::get_default_triple();
            Target::initialize_all(&InitializationConfig {
                asm_parser: true,
                asm_printer: true,
                base: true,
                disassembler: true,
                info: true,
                machine_code: true,
            });
            let target = Target::from_triple(&default_triplet).unwrap();
            let target_machine = target
                .create_target_machine(
                    &default_triplet,
                    TargetMachine::get_host_cpu_name().to_str().unwrap(),
                    TargetMachine::get_host_cpu_features().to_str().unwrap(),
                    OptimizationLevel::Aggressive,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .unwrap();

            let void_type = context.void_type();
            let char_type = context.i8_type();
            let int_type = context.i32_type();
            let size_type = context.ptr_sized_int_type(&target_machine.get_target_data(), None);
            let pointer_type = char_type.ptr_type(AddressSpace::default());

            Self {
                context,
                builder,
                module,
                target_machine,
                void_type,
                char_type,
                int_type,
                size_type,
                pointer_type,
            }
        }

        pub(super) fn module(&self) -> &Module<'a> {
            &self.module
        }

        pub(super) fn verify(&self) -> anyhow::Result<(), EmitError> {
            self.module
                .verify()
                .map_err(|error| EmitError::ModuleVerificationFailed(error.to_string()))
        }

        pub(super) fn optimize(&self) {
            let pass_manager_builder = PassManagerBuilder::create();
            pass_manager_builder.set_optimization_level(OptimizationLevel::Aggressive);

            let module_pass_manager = PassManager::create(());
            pass_manager_builder.populate_module_pass_manager(&module_pass_manager);

            let optimized_module = module_pass_manager.run_on(&self.module);
            dbg!(optimized_module);

            let function_pass_manager = PassManager::create(&self.module);
            pass_manager_builder.populate_function_pass_manager(&function_pass_manager);

            // todo: optimize all functions individually
            //
            // let optimized_main_function = function_pass_manager.run_on(&main_function);
            // dbg!(optimized_main_function);
        }

        pub(super) fn emit_assembly(&self, filename: &Path) -> anyhow::Result<(), EmitError> {
            self.target_machine
                .write_to_file(&self.module, FileType::Assembly, filename)
                .map_err(|error| EmitError::FailedToWriteToFile {
                    filename: filename.to_path_buf(),
                    error_message: error.to_string(),
                })
        }

        pub(super) fn emit_object_file(&self, filename: &Path) -> anyhow::Result<(), EmitError> {
            self.target_machine
                .write_to_file(&self.module, FileType::Object, filename)
                .map_err(|error| EmitError::FailedToWriteToFile {
                    filename: filename.to_path_buf(),
                    error_message: error.to_string(),
                })
        }

        pub(super) fn emit_llvm_ir(&self, filename: &Path) -> anyhow::Result<(), EmitError> {
            self.module
                .print_to_file(filename)
                .map_err(|error| EmitError::FailedToWriteToFile {
                    filename: filename.to_path_buf(),
                    error_message: error.to_string(),
                })
        }
    }
}

macro_rules! get_type {
    ($state:expr, Void) => {
        $state.void_type
    };
    ($state:expr, Char) => {
        $state.char_type
    };
    ($state:expr, Int) => {
        $state.int_type
    };
    ($state:expr, Size) => {
        $state.size_type
    };
    ($state:expr, Pointer) => {
        $state.pointer_type
    };
}

macro_rules! add_function {
        ($state:expr, $name:literal, [$(DataType::$parameter_type:ident),*$(,)?], DataType::$return_type:ident, $linkage:expr) => {{
            let function_type = get_type!($state, $return_type).fn_type(&[$(get_type!($state, $parameter_type).into()),*], false);
            let function_value = $state.module()
                    .add_function($name, function_type, $linkage);
            Function::new(
                $state.context,
                &$state.builder,
                function_value
            )
        }};
    }

pub(crate) fn emit(program: &Program, arguments: &CommandLineArguments) -> anyhow::Result<PathBuf> {
    let module_name = arguments
        .input_filename
        .file_prefix()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();

    let context = Context::create();
    let state = State::new(&context, &module_name);

    let putchar = add_function!(
        &state,
        "putchar",
        [DataType::Int],
        DataType::Int,
        Some(Linkage::External)
    );
    let malloc = add_function!(
        &state,
        "malloc",
        [DataType::Size],
        DataType::Pointer,
        Some(Linkage::External)
    );
    let free = add_function!(
        &state,
        "free",
        [DataType::Pointer],
        DataType::Void,
        Some(Linkage::External)
    );
    let main_function = add_function!(&state, "main", [], DataType::Int, Some(Linkage::External));

    let builder = main_function.append_basic_block("entry");

    let answer = state.int_type.const_int(33, false);
    let size = state.size_type.const_int(1024, false);

    let mem_ptr = builder
        .build_direct_call(malloc.into(), &[size.into()], "mem_ptr")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_pointer_value();

    let index = state.size_type.const_int(5, false);

    let element_addr = unsafe {
        builder
            .build_gep(state.int_type, mem_ptr, &[index], "element_addr")
            .unwrap()
    };
    builder.build_store(element_addr, answer).unwrap();

    let return_value = builder
        .build_direct_call(
            putchar.into(),
            &[builder
                .build_load(state.int_type, element_addr, "loaded_val")
                .unwrap()
                .into()],
            "putchar_result",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    builder
        .build_direct_call(free.into(), &[mem_ptr.into()], "")
        .unwrap();

    let sum = builder.build_int_add(answer, return_value, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();

    state.verify()?;

    state.optimize();

    match arguments.emit_target() {
        EmitTarget::Assembly => {
            state.emit_assembly(&arguments.output_filename())?;
            Ok(arguments.output_filename().clone())
        }
        EmitTarget::ObjectFile | EmitTarget::Executable => {
            let filename = match arguments.only_compile_and_assemble {
                true => arguments.output_filename(),
                false => {
                    let mut result = arguments.output_filename().clone();
                    result.set_extension(object_file_extension());
                    result
                }
            };
            state.emit_object_file(&filename)?;
            Ok(filename)
        }
        EmitTarget::LlvmIr => {
            state.emit_llvm_ir(&arguments.output_filename())?;
            Ok(arguments.output_filename().clone())
        }
    }
}

#[cfg(target_os = "windows")]
fn object_file_extension() -> &'static str {
    "obj"
}

#[cfg(target_os = "linux")]
fn object_file_extension() -> &'static str {
    "o"
}
