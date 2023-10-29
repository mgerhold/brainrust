use std::path::Path;

use inkwell::context::Context;
use inkwell::module::Linkage;
use inkwell::passes::{PassManager, PassManagerBuilder};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::BasicMetadataTypeEnum;
use inkwell::{AddressSpace, OptimizationLevel};

use crate::program::Program;

pub(crate) fn emit(program: &Program) {
    let context = Context::create();
    let builder = context.create_builder();
    let module = context.create_module("brainrust");

    let pass_manager_builder = PassManagerBuilder::create();
    pass_manager_builder.set_optimization_level(OptimizationLevel::Aggressive);

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

    let function_type = int_type.fn_type(&[], false);

    let main_function = module.add_function("main", function_type, Some(Linkage::External));

    let entry_block = context.append_basic_block(main_function, "entry");
    builder.position_at_end(entry_block);

    let answer = int_type.const_int(33, false);

    let malloc_function_type = pointer_type.fn_type(&[size_type.into()], false);
    let malloc_function =
        module.add_function("malloc", malloc_function_type, Some(Linkage::External));
    let free_function_type = void_type.fn_type(&[pointer_type.into()], false);
    let free_function = module.add_function("free", free_function_type, Some(Linkage::External));

    let putchar_function_type =
        int_type.fn_type(&[BasicMetadataTypeEnum::IntType(int_type)], false);
    let putchar_function =
        module.add_function("putchar", putchar_function_type, Some(Linkage::External));

    const SIZE: u64 = 1024;
    let size = size_type.const_int(SIZE, false);

    let mem_ptr = builder
        .build_direct_call(malloc_function, &[size.into()], "mem_ptr")
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_pointer_value();

    let index = size_type.const_int(5, false);

    let element_addr = unsafe {
        builder
            .build_gep(int_type, mem_ptr, &[index], "element_addr")
            .unwrap()
    };
    builder.build_store(element_addr, answer).unwrap();

    let return_value = builder
        .build_direct_call(
            putchar_function,
            &[builder
                .build_load(int_type, element_addr, "loaded_val")
                .unwrap()
                .into()],
            "putchar_result",
        )
        .unwrap()
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    builder
        .build_direct_call(free_function, &[mem_ptr.into()], "")
        .unwrap();

    let sum = builder.build_int_add(answer, return_value, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();

    module.verify().unwrap();

    let module_pass_manager = PassManager::create(());
    pass_manager_builder.populate_module_pass_manager(&module_pass_manager);

    let function_pass_manager = PassManager::create(&module);
    pass_manager_builder.populate_function_pass_manager(&function_pass_manager);

    let optimized_main_function = function_pass_manager.run_on(&main_function);
    dbg!(optimized_main_function);

    let optimized_module = module_pass_manager.run_on(&module);
    dbg!(optimized_module);

    target_machine
        .write_to_file(&module, FileType::Assembly, Path::new("test.asm"))
        .unwrap();
    target_machine
        .write_to_file(&module, FileType::Object, Path::new("test.o"))
        .unwrap();

    println!("{}", module.print_to_string().to_str().unwrap());
}
