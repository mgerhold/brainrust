use crate::interpreter::interpret;
use crate::parser::Parser;

//mod emitter;
mod interpreter;
mod parser;
mod program;

fn main() {
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
    match parser.parse() {
        Ok(program) => {
            //println!("{program}");
            interpret(&program);
            //emit(&program);
        }
        Err(error) => eprint!("{error:?}"),
    }
}
