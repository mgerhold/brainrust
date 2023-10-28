use std::fmt::{Display, Formatter};

pub(crate) struct Program {
    statements: Vec<Statement>,
}

impl Program {
    pub(crate) fn new(statements: Vec<Statement>) -> Self {
        Self { statements }
    }

    pub(crate) fn statements(&self) -> &Vec<Statement> {
        &self.statements
    }

    fn fmt_indented(&self, f: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        const EMPTY: &str = "";
        write!(f, "{EMPTY:0$}", indentation)?;
        for statement in &self.statements {
            match statement {
                Statement::IncrementPointer => write!(f, ">"),
                Statement::DecrementPointer => write!(f, "<"),
                Statement::IncrementValue => write!(f, "+"),
                Statement::DecrementValue => write!(f, "-"),
                Statement::PutChar => write!(f, "."),
                Statement::GetChar => write!(f, ","),
                Statement::Loop(statements) => {
                    writeln!(f, "[")?;
                    Program {
                        statements: statements.clone(),
                    }
                    .fmt_indented(f, indentation + 2)?;
                    writeln!(f, "\n{EMPTY:0$}]", indentation)
                }
            }?;
        }
        Ok(())
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt_indented(f, 0)
    }
}

pub(crate) enum StatementConversionError {
    InsignificantChar,
    OpeningLoop,
    ClosingLoop,
}

#[derive(Debug, Clone)]
pub(crate) enum Statement {
    IncrementPointer,
    DecrementPointer,
    IncrementValue,
    DecrementValue,
    PutChar,
    GetChar,
    Loop(Vec<Statement>),
}

impl TryFrom<u8> for Statement {
    type Error = StatementConversionError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'>' => Ok(Statement::IncrementPointer),
            b'<' => Ok(Statement::DecrementPointer),
            b'+' => Ok(Statement::IncrementValue),
            b'-' => Ok(Statement::DecrementValue),
            b'.' => Ok(Statement::PutChar),
            b',' => Ok(Statement::GetChar),
            b'[' => Err(StatementConversionError::OpeningLoop),
            b']' => Err(StatementConversionError::ClosingLoop),
            _ => Err(StatementConversionError::InsignificantChar),
        }
    }
}
