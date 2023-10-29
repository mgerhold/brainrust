use crate::interpreter::state::State;
use crate::program::{Program, Statement};

mod state {
    use std::io::{stdin, Read};

    #[derive(Default)]
    pub(super) struct State {
        memory: Vec<u8>,
        memory_offset: usize,
        pointer_address: i64,
    }

    impl State {
        pub(super) fn increment_pointer(&mut self) {
            self.pointer_address += 1;
        }

        pub(super) fn decrement_pointer(&mut self) {
            self.pointer_address -= 1;
        }

        pub(super) fn increment_value(&mut self) {
            let index = self.checked_index();
            self.memory[index] = self.memory[index].wrapping_add(1);
        }

        pub(super) fn decrement_value(&mut self) {
            let index = self.checked_index();
            self.memory[index] = self.memory[index].wrapping_sub(1);
        }

        pub(super) fn put_char(&mut self) {
            let index = self.checked_index();
            print!("{}", self.memory[index] as char)
        }

        pub(super) fn get_char(&mut self) {
            let input = stdin().lock().bytes().next().unwrap().unwrap();
            let index = self.checked_index();
            self.memory[index] = input;
        }

        pub(super) fn read_value(&mut self) -> u8 {
            let index = self.checked_index();
            self.memory[index]
        }

        fn checked_index(&mut self) -> usize {
            self.ensure_sufficient_memory_size();
            self.current_address_to_index() as usize
        }

        fn current_address_to_index(&self) -> i64 {
            self.pointer_address + self.memory_offset as i64
        }

        fn ensure_sufficient_memory_size(&mut self) {
            let target_index = self.current_address_to_index();
            if target_index < 0 {
                let difference = (-target_index) as usize;
                self.memory_offset += difference;
                self.memory.resize(self.memory.len() + difference, b'\0');
                for i in (difference..self.memory.len()).rev() {
                    self.memory[i] = self.memory[i - 1];
                }
            } else if target_index as usize >= self.memory.len() {
                let difference = target_index as usize - self.memory.len() + 1;
                self.memory.resize(self.memory.len() + difference, b'\0');
            }
            debug_assert!(
                self.current_address_to_index() >= 0
                    && (self.current_address_to_index() as usize) < self.memory.len()
            );
        }
    }
}

fn interpret_statement(statement: &Statement, state: &mut State) {
    match statement {
        Statement::IncrementPointer => state.increment_pointer(),
        Statement::DecrementPointer => state.decrement_pointer(),
        Statement::IncrementValue => state.increment_value(),
        Statement::DecrementValue => state.decrement_value(),
        Statement::PutChar => state.put_char(),
        Statement::GetChar => state.get_char(),
        Statement::Loop(statements) => {
            while state.read_value() != 0 {
                interpret_block(statements, state);
            }
        }
    }
}

fn interpret_block(statements: &[Statement], state: &mut State) {
    for statement in statements {
        interpret_statement(statement, state);
    }
}

pub(crate) fn interpret(program: &Program) {
    let mut interpreter_state = State::default();
    interpret_block(program.statements(), &mut interpreter_state);
}
