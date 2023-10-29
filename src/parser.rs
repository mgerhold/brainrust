use crate::program::{Program, Statement, StatementConversionError};
use anyhow::Result;
use std::fmt::{Display, Formatter};
use thiserror::Error;

#[derive(Error, Debug)]
pub(crate) enum ParserError {
    ClosingLoop,
    LoopNotClosed,
    UnexpectedChar(u8),
    EndOfSource,
}

impl Display for ParserError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub(crate) struct Parser<'a> {
    source: &'a [u8],
    index: usize,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(source: &'a [u8]) -> Self {
        Self { source, index: 0 }
    }

    pub(crate) fn parse(mut self) -> Result<Program, ParserError> {
        let block = self.block()?;
        if self.is_at_end() {
            Ok(Program::new(block))
        } else {
            Err(ParserError::UnexpectedChar(self.current()))
        }
    }

    fn block(&mut self) -> Result<Vec<Statement>, ParserError> {
        let mut statements = Vec::new();
        loop {
            match self.statement() {
                Ok(statement) => statements.push(statement),
                Err(ParserError::ClosingLoop | ParserError::EndOfSource) => break,
                Err(error) => return Err(error),
            }
        }
        Ok(statements)
    }

    fn statement(&mut self) -> Result<Statement, ParserError> {
        while !self.is_at_end() {
            match self.current().try_into() {
                Ok(statement) => {
                    self.advance();
                    return Ok(statement);
                }
                Err(StatementConversionError::InsignificantChar) => {}
                Err(StatementConversionError::OpeningLoop) => return self.loop_(),
                Err(StatementConversionError::ClosingLoop) => return Err(ParserError::ClosingLoop),
            }
            self.advance();
        }
        Err(ParserError::EndOfSource)
    }

    fn loop_(&mut self) -> Result<Statement, ParserError> {
        debug_assert!(self.current() == b'[');
        self.advance();
        let block = self.block()?;
        if self.current() != b']' {
            Err(ParserError::LoopNotClosed)
        } else {
            self.advance();
            Ok(Statement::Loop(block))
        }
    }

    fn current(&self) -> u8 {
        if self.is_at_end() {
            b'\0'
        } else {
            self.source[self.index]
        }
    }

    fn advance(&mut self) {
        self.index += 1;
    }

    fn is_at_end(&self) -> bool {
        self.index >= self.source.len()
    }
}
