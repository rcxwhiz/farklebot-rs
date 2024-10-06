use rand::{Rng, RngCore};

pub struct Roller<const D: usize, const S: u8> {
    rolls: Vec<[u8; D]>,
}

impl<const D: usize, const S: u8> Roller<D, S> {
    pub fn new() -> Self {
        let mut rolls = Vec::with_capacity(S.pow(D as u32) as usize);

        let mut roll = [1u8; D];
        loop {
            rolls.push(roll);

            let mut i = D - 1;
            while i >= 0 {
                if roll[i] < S {
                    roll[i] += 1;
                    break;
                } else {
                    roll[i] = 1;
                    if i == 0 {
                        return Self { rolls };
                    }
                    i -= 1;
                }
            }
        }
    }

    pub fn get_roll(&self, rng: &mut dyn RngCore) -> [u8; D] {
        let index = rng.gen::<usize>() % self.rolls.len();
        self.rolls[index]
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use crate::dice::roller::Roller;

    #[test]
    fn initialize() {
        let roller = Roller::<3, 3>::new();
        for roll in &roller.rolls {
            println!("{:?}", roll);
        }
    }

    #[test]
    fn get_roll() {
        let mut rng = &mut thread_rng();
        let roller = Roller::<3, 6>::new();
        for _ in 0..10 {
            println!("{:?}", roller.get_roll(&mut rng));
        }
    }
}
