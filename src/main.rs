use std::io;
use std::path::Path;
use std::process::{Command, Output};

use anyhow::Result;

use crate::emitter::emit;
use crate::interpreter::interpret;
use crate::parser::Parser;

mod emitter;
mod interpreter;
mod parser;
mod program;

fn invoke_clang(input_file: &Path, output_file: &Path) -> io::Result<Output> {
    Command::new("clang")
        .args([
            "-o",
            output_file.to_str().unwrap(),
            input_file.to_str().unwrap(),
        ])
        .output()
}

fn main() -> Result<()> {
    let hello_world = b"++++++++++
 [
  >+++++++>++++++++++>+++>+<<<<-
 ]                       Schleife zur Vorbereitung der Textausgabe
 >++.                    Ausgabe von 'H'
 >+.                     Ausgabe von 'e'
 +++++++.                'l'
 .                       'l'
 +++.                    'o'
 >++.                    Leerzeichen
 <<+++++++++++++++.      'W'
 >.                      'o'
 +++.                    'r'
 ------.                 'l'
 --------.               'd'
 >+.                     '!'
 >.                      Zeilenvorschub
 +++.                    Wagenruecklauf";

    let parser = Parser::new(hello_world);
    let program = parser.parse()?;

    interpret(&program);
    emit(&program);
    invoke_clang(Path::new("test.o"), Path::new("test.exe"))?;

    Ok(())
}
