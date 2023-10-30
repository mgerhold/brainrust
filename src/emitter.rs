use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use inkwell::context::Context;
use inkwell::{IntPredicate, OptimizationLevel};
use thiserror::Error;

use crate::command_line_arguments::{CommandLineArguments, EmitTarget};
use crate::emitter::state::State;
use crate::program::Program;

#[derive(Error, Debug)]
pub(crate) enum EmitError {
    FailedToWriteToFile {
        filename: PathBuf,
        error_message: String,
    },
    ModuleVerificationFailed(String),
    FailedToGetFunctionParameter(u32),
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
            EmitError::FailedToGetFunctionParameter(n) => {
                write!(f, "failed to get nth function parameter with n = {n}")
            }
        }
    }
}

mod state {
    use std::collections::HashMap;
    use std::path::Path;

    use inkwell::basic_block::BasicBlock;
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::module::Linkage;
    use inkwell::module::Module;
    use inkwell::passes::{PassManager, PassManagerBuilder};
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };
    use inkwell::types::{BasicMetadataTypeEnum, BasicType, IntType, PointerType, VoidType};
    use inkwell::values::{BasicValueEnum, FunctionValue, IntValue};
    use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

    use crate::emitter::EmitError;

    /*macro_rules! get_type {
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
        ($context:expr, $name:literal, [$(DataType::$parameter_type:ident),*$(,)?], DataType::$return_type:ident, $linkage:expr) => {{
            let function_type = get_type!($state, $return_type).fn_type(&[$(get_type!($state, $parameter_type).into()),*], false);
            let function_value = $state.module()
                    .add_function($name, function_type, $linkage);
            Function::new(
                $context,
                &$state.builder,
                function_value
            )
        }};
    }*/

    #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
    enum FunctionDeclaration {
        Putchar,
        Calloc,
        Malloc,
        Free,
        Realloc,
        EnsureSufficientMemoryCapacity,
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
        declared_functions: HashMap<FunctionDeclaration, FunctionValue<'a>>,
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
                declared_functions: Default::default(),
            }
        }

        fn function(&self, function_declaration: FunctionDeclaration) -> FunctionValue<'a> {
            *self.declared_functions.get(&function_declaration).unwrap()
        }

        pub(super) fn module(&self) -> &Module<'a> {
            &self.module
        }

        pub(super) fn verify(&self) -> anyhow::Result<(), EmitError> {
            self.module
                .verify()
                .map_err(|error| EmitError::ModuleVerificationFailed(error.to_string()))
        }

        pub(super) fn optimize(&self, level: OptimizationLevel) {
            let pass_manager_builder = PassManagerBuilder::create();
            pass_manager_builder.set_optimization_level(level);

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

        fn create_function(
            &self,
            name: &str,
            parameter_types: &[BasicMetadataTypeEnum<'a>],
            return_type: Option<&dyn BasicType<'a>>,
            linkage: Option<Linkage>,
        ) -> FunctionValue<'a> {
            let function_type = match return_type {
                Some(return_type) => return_type.fn_type(parameter_types, false),
                None => self.void_type.fn_type(parameter_types, false),
            };
            self.module.add_function(name, function_type, linkage)
        }

        pub(super) fn declare_libc_functions(&mut self) {
            self.declared_functions.insert(
                FunctionDeclaration::Putchar,
                self.create_function(
                    "putchar",
                    &[self.int_type.into()],
                    Some(&self.int_type),
                    Some(Linkage::External),
                ),
            );

            self.declared_functions.insert(
                FunctionDeclaration::Malloc,
                self.create_function(
                    "malloc",
                    &[self.size_type.into()],
                    Some(&self.pointer_type),
                    Some(Linkage::External),
                ),
            );

            self.declared_functions.insert(
                FunctionDeclaration::Free,
                self.create_function(
                    "free",
                    &[self.pointer_type.into()],
                    None,
                    Some(Linkage::External),
                ),
            );

            self.declared_functions.insert(
                FunctionDeclaration::Calloc,
                self.create_function(
                    "calloc",
                    &[self.size_type.into(), self.size_type.into()],
                    Some(&self.pointer_type),
                    Some(Linkage::External),
                ),
            );

            self.declared_functions.insert(
                FunctionDeclaration::Realloc,
                self.create_function(
                    "realloc",
                    &[self.pointer_type.into(), self.size_type.into()],
                    Some(&self.pointer_type),
                    Some(Linkage::External),
                ),
            );
        }

        pub(super) fn generate_function_ensure_sufficient_memory_capacity(
            &mut self,
        ) -> anyhow::Result<(), EmitError> {
            /* void ensure_sufficient_memory_capacity(
                    void** memory_ptr_ptr,
                    size_t* capacity_ptr,
                    size_t target_capacity
               )
            */
            let ensure_sufficient_memory_capacity = self.create_function(
                "ensure_sufficient_memory_capacity",
                &[
                    self.pointer_type.into(),
                    self.pointer_type.into(),
                    self.size_type.into(),
                ],
                None,
                Some(Linkage::Internal),
            );

            self.declared_functions.insert(
                FunctionDeclaration::EnsureSufficientMemoryCapacity,
                ensure_sufficient_memory_capacity,
            );

            let memory_ptr_ptr = ensure_sufficient_memory_capacity
                .get_nth_param(0)
                .unwrap()
                .into_pointer_value();
            let capacity_ptr = ensure_sufficient_memory_capacity
                .get_nth_param(1)
                .unwrap()
                .into_pointer_value();
            let target_capacity = ensure_sufficient_memory_capacity
                .get_nth_param(2)
                .unwrap()
                .into_int_value();

            let entry = self
                .context
                .append_basic_block(ensure_sufficient_memory_capacity, "entry");
            self.builder.position_at_end(entry);

            let capacity = self
                .builder
                .build_load(self.size_type, capacity_ptr, "capacity")
                .unwrap()
                .into_int_value();

            let must_grow = self
                .builder
                .build_int_compare(IntPredicate::UGT, target_capacity, capacity, "must_grow")
                .unwrap();

            let new_capacity = self
                .builder
                .build_alloca(self.size_type, "new_capacity")
                .unwrap();
            self.branch(
                must_grow,
                |after_branch| {
                    let capacity_is_zero = self
                        .builder
                        .build_int_compare(
                            IntPredicate::EQ,
                            capacity,
                            self.size_type.const_zero(),
                            "capacity_is_zero",
                        )
                        .unwrap();

                    self.branch(
                        capacity_is_zero,
                        |after_branch| {
                            self.builder
                                .build_store(new_capacity, self.size_type.const_int(1, false))
                                .unwrap();
                            self.builder
                                .build_unconditional_branch(after_branch)
                                .unwrap();
                        },
                        |after_branch| {
                            self.builder
                                .build_store(
                                    new_capacity,
                                    self.builder
                                        .build_int_mul(
                                            capacity,
                                            self.size_type.const_int(2, false),
                                            "new_capacity",
                                        )
                                        .unwrap(),
                                )
                                .unwrap();
                            self.builder
                                .build_unconditional_branch(after_branch)
                                .unwrap();
                        },
                    );
                    self.builder
                        .build_unconditional_branch(after_branch)
                        .unwrap();
                },
                |_| {
                    self.builder.build_return(None).unwrap();
                },
            );
            let memory_ptr = self
                .builder
                .build_load(self.pointer_type, memory_ptr_ptr, "memory_ptr")
                .unwrap();
            let new_memory_ptr = self
                .builder
                .build_direct_call(
                    self.function(FunctionDeclaration::Realloc).into(),
                    &[
                        memory_ptr.into(),
                        self.builder
                            .build_load(self.size_type, new_capacity, "new_capacity_value")
                            .unwrap()
                            .into(),
                    ],
                    "new_memory_ptr",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_pointer_value();
            self.builder
                .build_store(memory_ptr_ptr, new_memory_ptr)
                .unwrap();
            self.builder
                .build_store(
                    capacity_ptr,
                    self.builder
                        .build_load(self.size_type, new_capacity, "new_capacity_value")
                        .unwrap(),
                )
                .unwrap();
            self.builder.build_return(None).unwrap();
            Ok(())
        }

        pub(super) fn generate_function_main(&self) -> anyhow::Result<(), EmitError> {
            let main =
                self.create_function("main", &[], Some(&self.int_type), Some(Linkage::External));

            let entry = self.context.append_basic_block(main, "entry");
            self.builder.position_at_end(entry);

            let memory = self
                .builder
                .build_alloca(self.pointer_type, "memory")
                .unwrap();
            self.builder
                .build_store(memory, self.pointer_type.const_zero())
                .unwrap();

            let size = self.builder.build_alloca(self.size_type, "size").unwrap();
            self.builder
                .build_store(size, self.size_type.const_zero())
                .unwrap();

            let capacity = self
                .builder
                .build_alloca(self.size_type, "capacity")
                .unwrap();
            self.builder
                .build_store(capacity, self.size_type.const_zero())
                .unwrap();

            /* void ensure_sufficient_memory_capacity(
                    void** memory_ptr_ptr,
                    size_t* capacity_ptr,
                    size_t target_capacity
               )
            */
            for _ in 0..40 {
                self.builder
                    .build_direct_call(
                        self.function(FunctionDeclaration::EnsureSufficientMemoryCapacity)
                            .into(),
                        &[
                            memory.into(),
                            capacity.into(),
                            self.size_type.const_int(1000000, false).into(),
                        ],
                        "",
                    )
                    .unwrap();
            }

            self.builder
                .build_direct_call(
                    self.function(FunctionDeclaration::Free).into(),
                    &[self
                        .builder
                        .build_load(self.pointer_type, memory, "memory_address")
                        .unwrap()
                        .into()],
                    "free_result",
                )
                .unwrap();

            self.builder
                .build_return(Some(&self.int_type.const_zero()))
                .unwrap();

            Ok(())
        }

        fn branch<ThenEmitter: FnOnce(BasicBlock), ElseEmitter: FnOnce(BasicBlock)>(
            &self,
            condition: IntValue<'a>,
            then_emitter: ThenEmitter,
            else_emitter: ElseEmitter,
        ) {
            let surrounding_function = self
                .builder
                .get_insert_block()
                .unwrap()
                .get_parent()
                .unwrap();
            let then_block = self
                .context
                .append_basic_block(surrounding_function, "then");
            let else_block = self
                .context
                .append_basic_block(surrounding_function, "else");
            let after_branch_block = self
                .context
                .append_basic_block(surrounding_function, "after_branch");
            self.builder
                .build_conditional_branch(condition, then_block, else_block)
                .unwrap();

            self.builder.position_at_end(then_block);
            then_emitter(after_branch_block);

            self.builder.position_at_end(else_block);
            else_emitter(after_branch_block);

            self.builder.position_at_end(after_branch_block);
        }
    }
}

pub(crate) fn emit(program: &Program, arguments: &CommandLineArguments) -> anyhow::Result<PathBuf> {
    let module_name = arguments
        .input_filename
        .file_prefix()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();

    let context = Context::create();
    let mut state = State::new(&context, &module_name);
    state.declare_libc_functions();
    state.generate_function_ensure_sufficient_memory_capacity()?;
    state.generate_function_main()?;

    state.verify()?;

    state.optimize(OptimizationLevel::None);

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
