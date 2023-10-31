use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use inkwell::context::Context;
use inkwell::OptimizationLevel;
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
    use inkwell::values::{BasicMetadataValueEnum, FunctionValue, IntValue};
    use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

    use crate::emitter::state::FunctionDeclaration::Memset;
    use crate::emitter::EmitError;
    use crate::program::{Program, Statement};

    trait TypeHolder<'a> {
        fn void(&self) -> VoidType<'a>;
        fn char(&self) -> IntType<'a>;
        fn int(&self) -> IntType<'a>;
        fn size(&self) -> IntType<'a>;
        fn pointer(&self) -> PointerType<'a>;
    }

    #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
    enum FunctionDeclaration {
        Putchar,
        Calloc,
        Malloc,
        Free,
        Realloc,
        EnsureSufficientMemoryCapacity,
        AddressToIndex,
        Memmove,
        Memset,
        MemDump,
        Printf,
        Read,
        Write,
        GetChar,
    }

    struct TypeContainer<'a> {
        void_type: VoidType<'a>,
        char_type: IntType<'a>,
        int_type: IntType<'a>,
        size_type: IntType<'a>,
        pointer_type: PointerType<'a>,
    }

    impl<'a> TypeHolder<'a> for TypeContainer<'a> {
        fn void(&self) -> VoidType<'a> {
            self.void_type
        }

        fn char(&self) -> IntType<'a> {
            self.char_type
        }

        fn int(&self) -> IntType<'a> {
            self.int_type
        }

        fn size(&self) -> IntType<'a> {
            self.size_type
        }

        fn pointer(&self) -> PointerType<'a> {
            self.pointer_type
        }
    }

    type Functions<'a> = HashMap<FunctionDeclaration, FunctionValue<'a>>;

    pub(super) struct State<'a> {
        pub(super) module: Module<'a>,
        target_machine: TargetMachine,
    }

    impl<'a> State<'a> {
        pub(super) fn new(context: &'a Context, module_name: &str, program: &Program) -> Self {
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
                    RelocMode::PIC,
                    CodeModel::Default,
                )
                .unwrap();

            let types = TypeContainer {
                void_type: context.void_type(),
                char_type: context.i8_type(),
                int_type: context.i32_type(),
                size_type: context.ptr_sized_int_type(&target_machine.get_target_data(), None),
                pointer_type: context.i8_type().ptr_type(AddressSpace::default()),
            };

            let mut functions = HashMap::new();

            Self::declare_libc_functions(&mut functions, &module, &types);
            Self::generate_function_mem_dump(context, &builder, &mut functions, &module, &types);
            Self::generate_function_address_to_index(
                context,
                &builder,
                &mut functions,
                &module,
                &types,
            );
            Self::generate_function_ensure_sufficient_memory_capacity(
                context,
                &builder,
                &mut functions,
                &module,
                &types,
            );
            Self::generate_function_read(context, &builder, &mut functions, &module, &types);
            Self::generate_function_write(context, &builder, &mut functions, &module, &types);

            let run = Self::create_function(
                "run",
                &[
                    types.pointer().into(), // address_ptr (size_t*)
                    types.pointer().into(), // memory_ptr_ptr (char**)
                    types.pointer().into(), // capacity_ptr (size_t*)
                    types.pointer().into(), // offset_ptr (size_t*)
                ],
                None,
                Some(Linkage::Internal),
                false,
                &module,
                &types,
            );
            let entry = context.append_basic_block(run, "entry");
            builder.position_at_end(entry);
            for statement in program.statements() {
                Self::emit_code_for_statement(
                    statement, context, &builder, &functions, &module, &types,
                );
            }
            builder.build_return(None).unwrap();

            Self::generate_function_main(run, context, &builder, &mut functions, &module, &types);

            Self {
                module,
                target_machine,
            }
        }

        fn function(
            function_declaration: FunctionDeclaration,
            functions: &Functions<'a>,
        ) -> FunctionValue<'a> {
            *functions.get(&function_declaration).unwrap()
        }

        pub(super) fn verify(&self) -> anyhow::Result<(), EmitError> {
            self.module
                .verify()
                .map_err(|error| EmitError::ModuleVerificationFailed(error.to_string()))
        }

        pub(super) fn optimize(&self, level: OptimizationLevel) {
            println!("optimization level: {level:?}");
            let pass_manager_builder = PassManagerBuilder::create();
            pass_manager_builder.set_optimization_level(level);

            let module_pass_manager = PassManager::create(());
            pass_manager_builder.populate_module_pass_manager(&module_pass_manager);
            #[cfg(any(
                feature = "llvm4-0",
                feature = "llvm5-0",
                feature = "llvm6-0",
                feature = "llvm7-0",
                feature = "llvm8-0",
                feature = "llvm9-0",
                feature = "llvm10-0",
                feature = "llvm11-0",
                feature = "llvm12-0",
                feature = "llvm13-0",
                feature = "llvm14-0"
            ))]
            module_pass_manager.add_argument_promotion_pass();
            module_pass_manager.add_constant_merge_pass();
            #[cfg(not(any(
                feature = "llvm4-0",
                feature = "llvm5-0",
                feature = "llvm6-0",
                feature = "llvm7-0",
                feature = "llvm8-0",
                feature = "llvm9-0"
            )))]
            module_pass_manager.add_merge_functions_pass();
            module_pass_manager.add_dead_arg_elimination_pass();
            module_pass_manager.add_function_attrs_pass();
            module_pass_manager.add_function_inlining_pass();
            module_pass_manager.add_always_inliner_pass();
            module_pass_manager.add_global_dce_pass();
            module_pass_manager.add_global_optimizer_pass();
            #[cfg(not(any(
                feature = "llvm12-0",
                feature = "llvm13-0",
                feature = "llvm14-0",
                feature = "llvm15-0",
                feature = "llvm16-0"
            )))]
            //module_pass_manager.add_ip_constant_propagation_pass();
            /*#[cfg(not(feature = "llvm16-0"))]
            module_pass_manager.add_prune_eh_pass();*/
            module_pass_manager.add_ipsccp_pass();
            module_pass_manager.add_internalize_pass(true);
            module_pass_manager.add_strip_dead_prototypes_pass();
            module_pass_manager.add_strip_symbol_pass();
            #[cfg(feature = "llvm4-0")]
            module_pass_manager.add_bb_vectorize_pass();
            module_pass_manager.add_loop_vectorize_pass();
            module_pass_manager.add_slp_vectorize_pass();
            module_pass_manager.add_aggressive_dce_pass();
            module_pass_manager.add_bit_tracking_dce_pass();
            module_pass_manager.add_alignment_from_assumptions_pass();
            module_pass_manager.add_cfg_simplification_pass();
            module_pass_manager.add_dead_store_elimination_pass();
            module_pass_manager.add_scalarizer_pass();
            module_pass_manager.add_merged_load_store_motion_pass();
            module_pass_manager.add_gvn_pass();
            module_pass_manager.add_ind_var_simplify_pass();
            module_pass_manager.add_instruction_combining_pass();
            module_pass_manager.add_jump_threading_pass();
            module_pass_manager.add_licm_pass();
            module_pass_manager.add_loop_deletion_pass();
            module_pass_manager.add_loop_idiom_pass();
            module_pass_manager.add_loop_rotate_pass();
            module_pass_manager.add_loop_reroll_pass();
            module_pass_manager.add_loop_unroll_pass();
            #[cfg(any(
                feature = "llvm4-0",
                feature = "llvm5-0",
                feature = "llvm6-0",
                feature = "llvm7-0",
                feature = "llvm8-0",
                feature = "llvm9-0",
                feature = "llvm10-0",
                feature = "llvm11-0",
                feature = "llvm12-0",
                feature = "llvm13-0",
                feature = "llvm14-0"
            ))]
            module_pass_manager.add_loop_unswitch_pass();
            module_pass_manager.add_memcpy_optimize_pass();
            module_pass_manager.add_partially_inline_lib_calls_pass();
            module_pass_manager.add_lower_switch_pass();
            module_pass_manager.add_promote_memory_to_register_pass();
            module_pass_manager.add_reassociate_pass();
            module_pass_manager.add_sccp_pass();
            module_pass_manager.add_scalar_repl_aggregates_pass();
            module_pass_manager.add_scalar_repl_aggregates_pass_ssa();
            module_pass_manager.add_scalar_repl_aggregates_pass_with_threshold(1);
            module_pass_manager.add_simplify_lib_calls_pass();
            module_pass_manager.add_tail_call_elimination_pass();
            /*#[cfg(not(any(
                feature = "llvm12-0",
                feature = "llvm13-0",
                feature = "llvm14-0",
                feature = "llvm15-0",
                feature = "llvm16-0"
            )))]
            module_pass_manager.add_constant_propagation_pass();*/
            #[cfg(any(
                feature = "llvm12-0",
                feature = "llvm13-0",
                feature = "llvm14-0",
                feature = "llvm15-0",
                feature = "llvm16-0"
            ))]
            module_pass_manager.add_instruction_simplify_pass();
            module_pass_manager.add_demote_memory_to_register_pass();
            module_pass_manager.add_verifier_pass();
            module_pass_manager.add_correlated_value_propagation_pass();
            module_pass_manager.add_early_cse_pass();
            module_pass_manager.add_lower_expect_intrinsic_pass();
            module_pass_manager.add_type_based_alias_analysis_pass();
            module_pass_manager.add_scoped_no_alias_aa_pass();
            module_pass_manager.add_basic_alias_analysis_pass();
            module_pass_manager.add_early_cse_mem_ssa_pass();
            module_pass_manager.add_new_gvn_pass();

            /*#[cfg(not(any(
                feature = "llvm4-0",
                feature = "llvm5-0",
                feature = "llvm6-0",
                feature = "llvm16-0"
            )))]
            module_pass_manager.add_aggressive_inst_combiner_pass();*/
            #[cfg(not(any(feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
            module_pass_manager.add_loop_unroll_and_jam_pass();

            /*#[cfg(not(any(
                feature = "llvm4-0",
                feature = "llvm5-0",
                feature = "llvm6-0",
                feature = "llvm7-0",
                feature = "llvm15-0",
                feature = "llvm16-0"
            )))]
            {
                module_pass_manager.add_coroutine_early_pass();
                module_pass_manager.add_coroutine_split_pass();
                module_pass_manager.add_coroutine_elide_pass();
                module_pass_manager.add_coroutine_cleanup_pass();
            }*/

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
            name: &str,
            parameter_types: &[BasicMetadataTypeEnum<'a>],
            return_type: Option<&dyn BasicType<'a>>,
            linkage: Option<Linkage>,
            is_var_args: bool,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) -> FunctionValue<'a> {
            let function_type = match return_type {
                Some(return_type) => return_type.fn_type(parameter_types, is_var_args),
                None => type_holder.void().fn_type(parameter_types, is_var_args),
            };
            module.add_function(name, function_type, linkage)
        }

        fn declare_libc_functions(
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            functions.insert(
                FunctionDeclaration::GetChar,
                Self::create_function(
                    "getchar",
                    &[],
                    Some(&type_holder.int()),
                    Some(Linkage::External),
                    false,
                    module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Putchar,
                Self::create_function(
                    "putchar",
                    &[type_holder.int().into()],
                    Some(&type_holder.int()),
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Malloc,
                Self::create_function(
                    "malloc",
                    &[type_holder.size().into()],
                    Some(&type_holder.pointer()),
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Free,
                Self::create_function(
                    "free",
                    &[type_holder.pointer().into()],
                    None,
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Calloc,
                Self::create_function(
                    "calloc",
                    &[type_holder.size().into(), type_holder.size().into()],
                    Some(&type_holder.pointer()),
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Realloc,
                Self::create_function(
                    "realloc",
                    &[type_holder.pointer().into(), type_holder.size().into()],
                    Some(&type_holder.pointer()),
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Memmove,
                Self::create_function(
                    "memmove",
                    &[
                        type_holder.pointer().into(),
                        type_holder.pointer().into(),
                        type_holder.size().into(),
                    ],
                    Some(&type_holder.pointer()),
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Memset,
                Self::create_function(
                    "memset",
                    &[
                        type_holder.pointer().into(),
                        type_holder.int().into(),
                        type_holder.size().into(),
                    ],
                    Some(&type_holder.pointer()),
                    Some(Linkage::External),
                    false,
                    &module,
                    type_holder,
                ),
            );

            functions.insert(
                FunctionDeclaration::Printf,
                Self::create_function(
                    "printf",
                    &[type_holder.pointer().into()],
                    Some(&type_holder.int()),
                    Some(Linkage::External),
                    true,
                    &module,
                    type_holder,
                ),
            );
        }

        fn generate_function_address_to_index(
            context: &Context,
            builder: &Builder<'a>,
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            /* int64_t address_to_index(size_t offset, int64_t address) {
                   return offset + (size_t)address;
               }
            */
            let address_to_index = Self::create_function(
                "address_to_index",
                &[type_holder.size().into(), type_holder.size().into()],
                Some(&type_holder.size()),
                Some(Linkage::Internal),
                false,
                &module,
                type_holder,
            );

            functions.insert(FunctionDeclaration::AddressToIndex, address_to_index);

            let offset = address_to_index.get_nth_param(0).unwrap().into_int_value();
            let address = address_to_index.get_nth_param(1).unwrap().into_int_value();

            let entry = context.append_basic_block(address_to_index, "entry");
            builder.position_at_end(entry);

            let sum = builder.build_int_add(offset, address, "sum").unwrap();

            builder.build_return(Some(&sum)).unwrap();
        }

        fn generate_function_write(
            context: &Context,
            builder: &Builder<'a>,
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            /*
            void write(
                int64_t address,
                char value,
                char** memory_ptr_ptr,
                size_t* capacity_ptr,
                size_t* offset_ptr
            )
             */
            let write = Self::create_function(
                "write",
                &[
                    type_holder.size().into(),
                    type_holder.char().into(),
                    type_holder.pointer().into(),
                    type_holder.pointer().into(),
                    type_holder.pointer().into(),
                ],
                None,
                Some(Linkage::Internal),
                false,
                module,
                type_holder,
            );

            functions.insert(FunctionDeclaration::Write, write);

            let address = write.get_nth_param(0).unwrap().into_int_value();
            let value = write.get_nth_param(1).unwrap().into_int_value();
            let memory_ptr_ptr = write.get_nth_param(2).unwrap().into_pointer_value();
            let capacity_ptr = write.get_nth_param(3).unwrap().into_pointer_value();
            let offset_ptr = write.get_nth_param(4).unwrap().into_pointer_value();

            let entry = context.append_basic_block(write, "entry");
            builder.position_at_end(entry);

            builder
                .build_direct_call(
                    Self::function(
                        FunctionDeclaration::EnsureSufficientMemoryCapacity,
                        functions,
                    ),
                    &[
                        memory_ptr_ptr.into(),
                        capacity_ptr.into(),
                        offset_ptr.into(),
                        address.into(),
                    ],
                    "",
                )
                .unwrap();

            let index = builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::AddressToIndex, functions),
                    &[
                        address.into(),
                        builder
                            .build_load(type_holder.size(), offset_ptr, "offset")
                            .unwrap()
                            .into_int_value()
                            .into(),
                    ],
                    "index",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();

            let memory_address = unsafe {
                builder.build_gep(
                    type_holder.char(),
                    builder
                        .build_load(type_holder.pointer(), memory_ptr_ptr, "memory_ptr")
                        .unwrap()
                        .into_pointer_value(),
                    &[index],
                    "memory_address",
                )
            }
            .unwrap();

            builder.build_store(memory_address, value).unwrap();

            builder.build_return(None).unwrap();
        }

        fn generate_function_read(
            context: &Context,
            builder: &Builder<'a>,
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            /*
            char read(
                int64_t address,
                char** memory_ptr_ptr,
                size_t* capacity_ptr,
                size_t* offset_ptr
            )
             */
            let read = Self::create_function(
                "read",
                &[
                    type_holder.size().into(),
                    type_holder.pointer().into(),
                    type_holder.pointer().into(),
                    type_holder.pointer().into(),
                ],
                Some(&type_holder.char()),
                Some(Linkage::Internal),
                false,
                module,
                type_holder,
            );

            functions.insert(FunctionDeclaration::Read, read);

            let address = read.get_nth_param(0).unwrap().into_int_value();
            let memory_ptr_ptr = read.get_nth_param(1).unwrap().into_pointer_value();
            let capacity_ptr = read.get_nth_param(2).unwrap().into_pointer_value();
            let offset_ptr = read.get_nth_param(3).unwrap().into_pointer_value();

            let entry = context.append_basic_block(read, "entry");
            builder.position_at_end(entry);

            builder
                .build_direct_call(
                    Self::function(
                        FunctionDeclaration::EnsureSufficientMemoryCapacity,
                        functions,
                    ),
                    &[
                        memory_ptr_ptr.into(),
                        capacity_ptr.into(),
                        offset_ptr.into(),
                        address.into(),
                    ],
                    "",
                )
                .unwrap();

            builder
                .build_direct_call(
                    Self::function(
                        FunctionDeclaration::EnsureSufficientMemoryCapacity,
                        functions,
                    ),
                    &[
                        memory_ptr_ptr.into(),
                        capacity_ptr.into(),
                        offset_ptr.into(),
                        address.into(),
                    ],
                    "",
                )
                .unwrap();

            let index = builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::AddressToIndex, functions),
                    &[
                        address.into(),
                        builder
                            .build_load(type_holder.size(), offset_ptr, "offset")
                            .unwrap()
                            .into_int_value()
                            .into(),
                    ],
                    "index",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();

            let memory_address = unsafe {
                builder.build_gep(
                    type_holder.char(),
                    builder
                        .build_load(type_holder.pointer(), memory_ptr_ptr, "memory_ptr")
                        .unwrap()
                        .into_pointer_value(),
                    &[index],
                    "memory_address",
                )
            }
            .unwrap();

            let result = builder
                .build_load(type_holder.char(), memory_address, "result")
                .unwrap();

            builder.build_return(Some(&result)).unwrap();
        }

        fn generate_function_ensure_sufficient_memory_capacity(
            context: &'a Context,
            builder: &Builder<'a>,
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            /* void ensure_sufficient_memory_capacity(
                    char** memory_ptr_ptr,
                    size_t* capacity_ptr,
                    size_t* offset_ptr,
                    int64_t address
               )
            */
            let ensure_sufficient_memory_capacity = Self::create_function(
                "ensure_sufficient_memory_capacity",
                &[
                    type_holder.pointer().into(), // memory_ptr_ptr
                    type_holder.pointer().into(), // capacity_ptr
                    type_holder.pointer().into(), // offset_ptr
                    type_holder.size().into(),    // address
                ],
                None,
                Some(Linkage::Internal),
                false,
                &module,
                type_holder,
            );

            functions.insert(
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
            let offset_ptr = ensure_sufficient_memory_capacity
                .get_nth_param(2)
                .unwrap()
                .into_pointer_value();
            let address = ensure_sufficient_memory_capacity
                .get_nth_param(3)
                .unwrap()
                .into_int_value();

            let entry = context.append_basic_block(ensure_sufficient_memory_capacity, "entry");
            builder.position_at_end(entry);

            let index = builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::AddressToIndex, functions),
                    &[
                        builder
                            .build_load(type_holder.size(), offset_ptr, "offset")
                            .unwrap()
                            .into(),
                        address.into(),
                    ],
                    "index",
                )
                .unwrap()
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();

            let is_index_negative = builder
                .build_int_compare(
                    IntPredicate::SLT,
                    index,
                    type_holder.size().const_zero(),
                    "is_index_negative",
                )
                .unwrap();

            Self::branch(
                context,
                builder,
                is_index_negative,
                |after_branch| {
                    // the index is negative

                    // size_t difference = (size_t)(-index);
                    let difference = builder.build_int_neg(index, "difference").unwrap();
                    // (*offset) += difference;
                    builder
                        .build_store(
                            offset_ptr,
                            builder
                                .build_int_add(
                                    builder
                                        .build_load(type_holder.size(), offset_ptr, "offset")
                                        .unwrap()
                                        .into_int_value(),
                                    difference,
                                    "new_offset",
                                )
                                .unwrap(),
                        )
                        .unwrap();

                    // size_t new_capacity = *capacity_ptr + difference;
                    let new_capacity = builder
                        .build_int_add(
                            builder
                                .build_load(type_holder.size(), capacity_ptr, "capacity")
                                .unwrap()
                                .into_int_value(),
                            difference,
                            "new_capacity",
                        )
                        .unwrap();

                    // char* new_memory_ptr = malloc(*memory_ptr_ptr, new_capacity);
                    let new_memory_ptr = builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Realloc, functions),
                            &[
                                builder
                                    .build_load(type_holder.pointer(), memory_ptr_ptr, "memory_ptr")
                                    .unwrap()
                                    .into_pointer_value()
                                    .into(),
                                new_capacity.into(),
                            ],
                            "new_memory_ptr",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_pointer_value();

                    // char* dest = &new_memory_ptr[difference];
                    let dest = unsafe {
                        builder
                            .build_gep(type_holder.char(), new_memory_ptr, &[difference], "dest")
                            .unwrap()
                    };

                    // memmove(dest, new_memory_ptr, *capacity_ptr)
                    builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Memmove, functions),
                            &[
                                dest.into(),
                                new_memory_ptr.into(),
                                builder
                                    .build_load(type_holder.size(), capacity_ptr, "capacity")
                                    .unwrap()
                                    .into(),
                            ],
                            "",
                        )
                        .unwrap();

                    // memset(new_memory_ptr, 0, difference)
                    builder
                        .build_direct_call(
                            Self::function(Memset, functions),
                            &[
                                new_memory_ptr.into(),
                                type_holder.int().const_int(0, false).into(),
                                difference.into(),
                            ],
                            "",
                        )
                        .unwrap();

                    // *capacity_ptr = new_capacity;
                    builder.build_store(capacity_ptr, new_capacity).unwrap();

                    // *memory_ptr_ptr = new_memory_ptr;
                    builder.build_store(memory_ptr_ptr, new_memory_ptr).unwrap();

                    builder.build_unconditional_branch(after_branch).unwrap();
                },
                |after_branch| {
                    let index_is_greater_than_or_equal_to_capacity = builder
                        .build_int_compare(
                            IntPredicate::UGE,
                            index,
                            builder
                                .build_load(type_holder.size(), capacity_ptr, "capacity")
                                .unwrap()
                                .into_int_value(),
                            "index_is_greater_than_or_equal_to_capacity",
                        )
                        .unwrap();

                    Self::branch(
                        context,
                        builder,
                        index_is_greater_than_or_equal_to_capacity,
                        |after_branch| {
                            // size_t new_capacity = index + 1;
                            let new_capacity = builder
                                .build_int_add(
                                    index,
                                    type_holder.size().const_int(1, false),
                                    "new_capacity",
                                )
                                .unwrap();

                            // char* new_memory_ptr = realloc(memory_ptr, new_capacity);
                            let new_memory_ptr = builder
                                .build_direct_call(
                                    Self::function(FunctionDeclaration::Realloc, functions),
                                    &[
                                        builder
                                            .build_load(
                                                type_holder.pointer(),
                                                memory_ptr_ptr,
                                                "memory_ptr",
                                            )
                                            .unwrap()
                                            .into_pointer_value()
                                            .into(),
                                        new_capacity.into(),
                                    ],
                                    "new_memory_ptr",
                                )
                                .unwrap()
                                .try_as_basic_value()
                                .unwrap_left()
                                .into_pointer_value();

                            // size_t difference = new_capacity - capacity;
                            let difference = builder
                                .build_int_sub(
                                    new_capacity,
                                    builder
                                        .build_load(type_holder.size(), capacity_ptr, "capacity")
                                        .unwrap()
                                        .into_int_value(),
                                    "difference",
                                )
                                .unwrap();

                            // char* dest = &new_memory_ptr[*capacity_ptr];
                            let dest = unsafe {
                                builder
                                    .build_gep(
                                        type_holder.char(),
                                        new_memory_ptr,
                                        &[builder
                                            .build_load(
                                                type_holder.size(),
                                                capacity_ptr,
                                                "capacity",
                                            )
                                            .unwrap()
                                            .into_int_value()],
                                        "dest",
                                    )
                                    .unwrap()
                            };

                            // memset(dest, 0, difference);
                            builder
                                .build_direct_call(
                                    Self::function(FunctionDeclaration::Memset, functions),
                                    &[
                                        dest.into(),
                                        type_holder.int().const_int(0, false).into(),
                                        difference.into(),
                                    ],
                                    "",
                                )
                                .unwrap();

                            // *memory_ptr_ptr = new_memory_ptr;
                            builder.build_store(memory_ptr_ptr, new_memory_ptr).unwrap();

                            // *capacity_ptr = new_capacity;
                            builder.build_store(capacity_ptr, new_capacity).unwrap();

                            builder.build_unconditional_branch(after_branch).unwrap();
                        },
                        |_| {
                            builder.build_return(None).unwrap();
                        },
                    );

                    builder.build_unconditional_branch(after_branch).unwrap();
                },
            );

            builder.build_return(None).unwrap();
        }

        fn generate_function_mem_dump(
            context: &'a Context,
            builder: &Builder<'a>,
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            let mem_dump = Self::create_function(
                "mem_dump",
                &[type_holder.pointer().into(), type_holder.size().into()],
                None,
                Some(Linkage::Internal),
                false,
                module,
                type_holder,
            );

            functions.insert(FunctionDeclaration::MemDump, mem_dump);

            let memory_ptr = mem_dump.get_nth_param(0).unwrap().into_pointer_value();
            let capacity = mem_dump.get_nth_param(1).unwrap().into_int_value();

            let entry = context.append_basic_block(mem_dump, "entry");
            let loop_start = context.append_basic_block(mem_dump, "loop_start");
            let loop_body = context.append_basic_block(mem_dump, "loop_body");
            let after_loop = context.append_basic_block(mem_dump, "after_loop");
            builder.position_at_end(entry);

            builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::Putchar, functions),
                    &[type_holder.int().const_int(77, false).into()],
                    "",
                )
                .unwrap();

            builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::Putchar, functions),
                    &[type_holder.int().const_int(58, false).into()],
                    "",
                )
                .unwrap();

            let i_ptr = builder.build_alloca(type_holder.size(), "i").unwrap();
            builder
                .build_store(i_ptr, type_holder.size().const_zero())
                .unwrap();

            builder.build_unconditional_branch(loop_start).unwrap();

            builder.position_at_end(loop_start);

            let i_is_less_than_capacity = builder
                .build_int_compare(
                    IntPredicate::ULT,
                    builder
                        .build_load(type_holder.size(), i_ptr, "i")
                        .unwrap()
                        .into_int_value(),
                    capacity,
                    "i_is_less_than_capacity",
                )
                .unwrap();

            Self::branch(
                context,
                builder,
                i_is_less_than_capacity,
                |after_branch| {
                    builder.build_unconditional_branch(after_branch).unwrap();
                },
                |_| {
                    builder.build_return(None).unwrap();
                },
            );
            builder.build_unconditional_branch(loop_body).unwrap();

            builder.position_at_end(loop_body);
            // char* address = &memory_ptr[i]
            let address = unsafe {
                builder
                    .build_gep(
                        type_holder.char(),
                        memory_ptr,
                        &[builder
                            .build_load(type_holder.size(), i_ptr, "i")
                            .unwrap()
                            .into_int_value()],
                        "address",
                    )
                    .unwrap()
            };
            let value = builder
                .build_load(type_holder.char(), address, "value")
                .unwrap()
                .into_int_value();

            let printable_value = builder
                .build_int_add(
                    value,
                    type_holder.char().const_int(48, false),
                    "printable_value",
                )
                .unwrap();

            let printable_value_int = builder
                .build_int_cast(printable_value, type_holder.int(), "printable_value_int")
                .unwrap();

            builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::Putchar, functions),
                    &[printable_value_int.into()],
                    "",
                )
                .unwrap();
            let new_i = builder
                .build_int_add(
                    builder
                        .build_load(type_holder.size(), i_ptr, "i")
                        .unwrap()
                        .into_int_value(),
                    type_holder.size().const_int(1, false),
                    "new_i",
                )
                .unwrap();
            builder.build_store(i_ptr, new_i).unwrap();
            builder.build_unconditional_branch(loop_start).unwrap();

            builder.position_at_end(after_loop);
            builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::Putchar, functions),
                    &[type_holder.int().const_int(10, false).into()],
                    "",
                )
                .unwrap();
            builder.build_return(None).unwrap();
        }

        #[allow(dead_code)]
        fn generate_printf(
            format_string: &str,
            args: &[BasicMetadataValueEnum],
            builder: &Builder<'a>,
            functions: &Functions<'a>,
        ) {
            let string = unsafe {
                builder
                    .build_global_string(format_string, "string")
                    .unwrap()
            };
            let first: &[BasicMetadataValueEnum] = &[string.as_pointer_value().into()];
            builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::Printf, functions),
                    &[first, args].concat(),
                    "",
                )
                .unwrap();
        }

        fn generate_function_main(
            run_function: FunctionValue<'a>,
            context: &Context,
            builder: &Builder<'a>,
            functions: &mut Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            let main = Self::create_function(
                "main",
                &[],
                Some(&type_holder.int()),
                Some(Linkage::External),
                false,
                &module,
                type_holder,
            );

            let entry = context.append_basic_block(main, "entry");
            builder.position_at_end(entry);

            let memory_ptr_ptr = builder
                .build_alloca(type_holder.pointer(), "memory")
                .unwrap();
            builder
                .build_store(memory_ptr_ptr, type_holder.pointer().const_zero())
                .unwrap();

            let capacity_ptr = builder
                .build_alloca(type_holder.size(), "capacity")
                .unwrap();
            builder
                .build_store(capacity_ptr, type_holder.size().const_zero())
                .unwrap();

            let offset_ptr = builder.build_alloca(type_holder.size(), "offset").unwrap();
            builder
                .build_store(offset_ptr, type_holder.size().const_zero())
                .unwrap();

            let address_ptr = builder.build_alloca(type_holder.size(), "address").unwrap();
            builder
                .build_store(address_ptr, type_holder.size().const_zero())
                .unwrap();

            let ensure_address = |index: i64| {
                builder
                    .build_direct_call(
                        Self::function(
                            FunctionDeclaration::EnsureSufficientMemoryCapacity,
                            functions,
                        ),
                        &[
                            memory_ptr_ptr.into(),
                            capacity_ptr.into(),
                            offset_ptr.into(),
                            if index < 0 {
                                builder
                                    .build_int_neg(
                                        type_holder.size().const_int((-index) as u64, false),
                                        "",
                                    )
                                    .unwrap()
                                    .into()
                            } else {
                                type_holder.size().const_int(index as u64, false).into()
                            },
                        ],
                        "",
                    )
                    .unwrap()
            };

            ensure_address(15000);
            ensure_address(-15000);

            builder
                .build_direct_call(
                    run_function,
                    &[
                        address_ptr.into(),
                        memory_ptr_ptr.into(),
                        capacity_ptr.into(),
                        offset_ptr.into(),
                    ],
                    "",
                )
                .unwrap();

            builder
                .build_direct_call(
                    Self::function(FunctionDeclaration::Free, functions),
                    &[builder
                        .build_load(type_holder.pointer(), memory_ptr_ptr, "memory_address")
                        .unwrap()
                        .into()],
                    "",
                )
                .unwrap();

            builder
                .build_return(Some(&type_holder.int().const_zero()))
                .unwrap();
        }

        fn branch<ThenEmitter: FnOnce(BasicBlock<'a>), ElseEmitter: FnOnce(BasicBlock<'a>)>(
            context: &'a Context,
            builder: &Builder<'a>,
            condition: IntValue<'a>,
            then_emitter: ThenEmitter,
            else_emitter: ElseEmitter,
        ) {
            let surrounding_function = builder.get_insert_block().unwrap().get_parent().unwrap();
            let then_block = context.append_basic_block(surrounding_function, "then");
            let else_block = context.append_basic_block(surrounding_function, "else");
            let after_branch_block =
                context.append_basic_block(surrounding_function, "after_branch");
            builder
                .build_conditional_branch(condition, then_block, else_block)
                .unwrap();

            builder.position_at_end(then_block);
            then_emitter(after_branch_block);

            builder.position_at_end(else_block);
            else_emitter(after_branch_block);

            builder.position_at_end(after_branch_block);
        }

        fn emit_code_for_statement(
            statement: &Statement,
            context: &'a Context,
            builder: &Builder<'a>,
            functions: &Functions<'a>,
            module: &Module<'a>,
            type_holder: &dyn TypeHolder<'a>,
        ) {
            /*
            types.pointer().into(), // address_ptr (size_t*)
            types.pointer().into(), // memory_ptr_ptr (char**)
            types.pointer().into(), // capacity_ptr (size_t*)
            types.pointer().into(), // offset_ptr (size_t*)
             */
            let current_function = builder.get_insert_block().unwrap().get_parent().unwrap();
            let address_ptr = current_function
                .get_nth_param(0)
                .unwrap()
                .into_pointer_value();
            let memory_ptr_ptr = current_function
                .get_nth_param(1)
                .unwrap()
                .into_pointer_value();
            let capacity_ptr = current_function
                .get_nth_param(2)
                .unwrap()
                .into_pointer_value();
            let offset_ptr = current_function
                .get_nth_param(3)
                .unwrap()
                .into_pointer_value();

            match statement {
                Statement::IncrementPointer => {
                    let address = builder
                        .build_load(type_holder.size(), address_ptr, "address")
                        .unwrap()
                        .into_int_value();
                    let incremented = builder
                        .build_int_add(
                            address,
                            type_holder.size().const_int(1, false),
                            "incremented",
                        )
                        .unwrap();
                    builder.build_store(address_ptr, incremented).unwrap();
                }
                Statement::DecrementPointer => {
                    let address = builder
                        .build_load(type_holder.size(), address_ptr, "address")
                        .unwrap()
                        .into_int_value();
                    let decremented = builder
                        .build_int_sub(
                            address,
                            type_holder.size().const_int(1, false),
                            "decremented",
                        )
                        .unwrap();
                    builder.build_store(address_ptr, decremented).unwrap();
                }
                Statement::IncrementValue => {
                    let value = builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Read, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();
                    let incremented = builder
                        .build_int_add(value, type_holder.char().const_int(1, false), "incremented")
                        .unwrap();

                    builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Write, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                incremented.into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "",
                        )
                        .unwrap();
                }
                Statement::DecrementValue => {
                    let value = builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Read, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();
                    let decremented = builder
                        .build_int_sub(value, type_holder.char().const_int(1, false), "decremented")
                        .unwrap();

                    builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Write, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                decremented.into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "",
                        )
                        .unwrap();
                }
                Statement::PutChar => {
                    let value = builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Read, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();

                    let int_value = builder
                        .build_int_cast(value, type_holder.int(), "int_value")
                        .unwrap();

                    builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Putchar, functions),
                            &[int_value.into()],
                            "",
                        )
                        .unwrap();
                }
                Statement::GetChar => {
                    let value = builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::GetChar, functions),
                            &[],
                            "value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();

                    let char_value = builder
                        .build_int_cast(value, type_holder.char(), "char_value")
                        .unwrap();

                    builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Write, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                char_value.into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "",
                        )
                        .unwrap();
                }
                Statement::Loop(statements) => {
                    let current_function =
                        builder.get_insert_block().unwrap().get_parent().unwrap();
                    let loop_start = context.append_basic_block(current_function, "loop_start");
                    let after_loop = context.append_basic_block(current_function, "after_loop");
                    builder.build_unconditional_branch(loop_start).unwrap();

                    builder.position_at_end(loop_start);
                    let value = builder
                        .build_direct_call(
                            Self::function(FunctionDeclaration::Read, functions),
                            &[
                                builder
                                    .build_load(type_holder.size(), address_ptr, "address")
                                    .unwrap()
                                    .into_int_value()
                                    .into(),
                                memory_ptr_ptr.into(),
                                capacity_ptr.into(),
                                offset_ptr.into(),
                            ],
                            "value",
                        )
                        .unwrap()
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value();

                    let condition = builder
                        .build_int_compare(
                            IntPredicate::NE,
                            value,
                            type_holder.char().const_zero(),
                            "condition",
                        )
                        .unwrap();

                    Self::branch(
                        context,
                        builder,
                        condition,
                        |_| {
                            for statement in statements {
                                Self::emit_code_for_statement(
                                    statement,
                                    context,
                                    builder,
                                    functions,
                                    module,
                                    type_holder,
                                );
                            }
                            builder.build_unconditional_branch(loop_start).unwrap();
                        },
                        |after_branch| {
                            builder.build_unconditional_branch(after_branch).unwrap();
                        },
                    );
                    builder.build_unconditional_branch(after_loop).unwrap();

                    builder.position_at_end(after_loop);
                }
            }
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
    let state = State::new(&context, &module_name, program);

    match state.verify() {
        Ok(_) => {}
        Err(error) => eprintln!("{error:?}"),
    }

    state.optimize(arguments.optimization_level());

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
