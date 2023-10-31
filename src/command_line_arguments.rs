use clap::ValueEnum;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmitTarget {
    Assembly,
    ObjectFile,
    Executable,
    LlvmIr,
}

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
#[clap(group(
            clap::ArgGroup::new("output")
                .required(false)
                .args(& ["interpret", "emit_assembly", "only_compile_and_assemble", "emit_llvm"])
        ))]
pub(crate) struct CommandLineArguments {
    #[arg(short, long, help = "Name of the file to be generated")]
    output_filename: Option<PathBuf>,

    #[arg(
        short = 'r',
        long = "run",
        action,
        help = "Interpret instead of compile"
    )]
    pub(crate) interpret: bool,

    #[arg(
        short = 'a',
        long = "assembly",
        action,
        help = "Emit assembler code only"
    )]
    pub(crate) emit_assembly: bool,

    #[arg(short = 'c', action, help = "Only run compile and assemble steps")]
    pub(crate) only_compile_and_assemble: bool,

    #[arg(
        long = "emit-llvm",
        action,
        help = "Emit LLVM intermediate representation"
    )]
    pub(crate) emit_llvm: bool,

    pub(crate) input_filename: PathBuf,

    #[arg(short = 'O', value_parser = clap::value_parser!(u8).range(0..=3), help = "Sets the optimization level", default_value_t = 2)]
    optimization_level: u8,
}

impl CommandLineArguments {
    pub(crate) fn emit_target(&self) -> EmitTarget {
        match (
            self.emit_assembly,
            self.only_compile_and_assemble,
            self.emit_llvm,
        ) {
            (true, false, false) => EmitTarget::Assembly,
            (false, true, false) => EmitTarget::ObjectFile,
            (false, false, true) => EmitTarget::LlvmIr,
            (false, false, false) => EmitTarget::Executable,
            _ => unreachable!(),
        }
    }

    pub(crate) fn optimization_level(&self) -> inkwell::OptimizationLevel {
        match self.optimization_level {
            0 => inkwell::OptimizationLevel::None,
            1 => inkwell::OptimizationLevel::Less,
            2 => inkwell::OptimizationLevel::Default,
            3 => inkwell::OptimizationLevel::Aggressive,
            _ => unreachable!("value was checked by clap"),
        }
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn output_filename(&self) -> PathBuf {
        self.output_filename.as_ref().cloned().unwrap_or_else(|| {
            PathBuf::from(match self.emit_target() {
                EmitTarget::Assembly => "out.asm",
                EmitTarget::ObjectFile => "out.obj",
                EmitTarget::Executable => "a.exe",
                EmitTarget::LlvmIr => "out.ll",
            })
        })
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn output_filename(&self) -> PathBuf {
        self.output_filename.as_ref().cloned().unwrap_or_else(|| {
            PathBuf::from(match self.emit_target() {
                EmitTarget::Assembly => "out.asm",
                EmitTarget::ObjectFile => "out.o",
                EmitTarget::Executable => "a.out",
                EmitTarget::LlvmIr => "out.ll",
            })
        })
    }
}
