#![feature(path_file_prefix)]

use std::io;
use std::path::Path;
use std::process::{Command, ExitStatus};

use crate::command_line_arguments::{CommandLineArguments, EmitTarget};
use anyhow::Result;
use clap::Parser as _;

use crate::emitter::emit;
use crate::interpreter::interpret;
use crate::parser::Parser;

mod emitter;
mod interpreter;
mod parser;
mod program;

mod command_line_arguments;

fn link(input_file: &Path, output_file: &Path) -> io::Result<ExitStatus> {
    Command::new("clang")
        .args([
            "-o",
            output_file.to_str().unwrap(),
            input_file.to_str().unwrap(),
        ])
        .status()
}

fn read_source(filename: &Path) -> io::Result<Vec<u8>> {
    std::fs::read(filename)
}

fn main() -> Result<()> {
    let command_line_arguments = CommandLineArguments::parse();

    let source = read_source(&command_line_arguments.input_filename)?;

    let parser = Parser::new(&source);
    let program = parser.parse()?;

    if command_line_arguments.interpret {
        interpret(&program);
    } else {
        let compiler_output_filename = emit(&program, &command_line_arguments)?;
        if command_line_arguments.emit_target() == EmitTarget::Executable {
            link(
                &compiler_output_filename,
                &command_line_arguments.output_filename(),
            )?;
        }
    }

    Ok(())
}
