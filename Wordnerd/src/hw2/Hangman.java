package hw2;

import java.util.Random;


public class Hangman extends Game{
	static final int MIN_WORD_LENGTH = 5; //minimum length of puzzle word
	static final int MAX_WORD_LENGTH = 10; //maximum length of puzzle word
	static final int HANGMAN_TRIALS = 10;  // max number of trials in a game
	static final int HANGMAN_GAME_TIME = 30; // max time in seconds for one round of game
	
	HangmanRound hangmanRound;
	
	/** setupRound() is a replacement of findPuzzleWord() in HW1. 
	 * It returns a new HangmanRound instance with puzzleWord initialized randomly drawn from wordsFromFile.
	* The puzzleWord must be a word between HANGMAN_MIN_WORD_LENGTH and HANGMAN_MAX_WORD_LEGTH. 
	* Other properties of Hangmanround are also initialized here. 
	*/
	@Override
	HangmanRound setupRound() {
		//write your code here
		hangmanRound = new HangmanRound();
		WordNerdModel w = new WordNerdModel();
		String[] wordString = w.getWordsFromFile();
		int totalWordNumer = wordString.length;	//find how many words in the file
		boolean flag = true;								//contain if the random word in the required word length range
		String puzzleWord = "";
		while (flag) {
			puzzleWord = wordString[random(totalWordNumer)];
//					Hangman.wordsFromFile[this.random(totalWordNumer)];	//random a puzzle word from the file.
			//check if the word within the range
			if (puzzleWord.length()>MAX_WORD_LENGTH || puzzleWord.length()< MIN_WORD_LENGTH) {
				flag = true;
			}
			else {
				flag = false;
			}
		}
		hangmanRound.setPuzzleWord(puzzleWord);
		return hangmanRound;
//		return puzzleWord;
//		return null;
	}
	
	
	/** Returns a clue that has at least half the number of letters in puzzleWord replaced with dashes.
	* The replacement should stop as soon as number of dashes equals or exceeds 50% of total word length. 
	* Note that repeating letters will need to be replaced together.
	* For example, in 'apple', if replacing p, then both 'p's need to be replaced to make it a--le */
	@Override
	String makeAClue(String puzzleWord) {
		//write your code here
		String clueWord = puzzleWord;
		char[] charCW = clueWord.toCharArray();//change the clue word to a char array
		int maxDashes = (int)Math.ceil((double)charCW.length/2);// max dashes can have in the puzzle word
		//if the clue word does not have enough dashes, make it to the max dashes
		while (this.countDashes(clueWord)<maxDashes) {
			int randomNumber = this.random(charCW.length);//random a number
			char replaceChar = charCW[randomNumber];//find the char need to be replaced
			//replace all the char by dashes
			for (int i = 0; i < charCW.length; i++) {
				if (charCW[i]==replaceChar) {
					charCW[i] = '-';
				}
			}
			clueWord = String.valueOf(charCW);
		}
		return clueWord;
	}

	/** countDashes() returns the number of dashes in a clue String */ 
	int countDashes(String word) {
		//write your code here
		return 0;
	}
	
	/** getScoreString() returns a formatted String with calculated score to be displayed after
	 * each trial in Hangman. See the handout and the video clips for specific format of the string. */
	@Override
	String getScoreString() {
		//write your code here
		return null;
	}

	/** nextTry() takes next guess and updates hitCount, missCount, and clueWord in hangmanRound. 
	* Returns INDEX for one of the images defined in GameView (SMILEY_INDEX, THUMBS_UP_INDEX...etc. 
	* The key change from HW1 is that because the keyboardButtons will be disabled after the player clicks on them, 
	* there is no need to track the previous guesses made in userInputs*/
	@Override
	int nextTry(String guess) {  
		//write your code here
		return 0;
	}
	
	private int random(int range) {
		Random random = new Random();
		int randomNumber = random.nextInt(range);	//random a number from 0 to range
		return randomNumber;
	}
	
	//TEST
//	public static void main(String[] args) {
//		Hangman h = new Hangman();
//		h.setupRound();
//	}
}

