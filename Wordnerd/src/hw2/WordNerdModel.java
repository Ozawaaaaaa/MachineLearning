package hw2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class WordNerdModel {
	
	static String[] wordsFromFile;
	static final String WORDS_FILE_NAME= "data/wordsFile.txt";
	
	static void readWordsFile(String wordsFilename) {
		File file = new File(wordsFilename);
		Scanner inputs;
		try {
			inputs = new Scanner(file);
			StringBuilder readin = new StringBuilder();
			while (inputs.hasNextLine()) {
				readin.append(inputs.nextLine()+"\n");
			}
			wordsFromFile= readin.toString().trim().split("\n");//create a string array to store all the words read from the file
			inputs.close();
		} catch (FileNotFoundException e) {System.out.println("Sorry, we cannot find the file");}
	}
	
	public String[] getWordsFromFile() {return WordNerdModel.wordsFromFile;}
	
	//TEST
//	public static void main(String[] args) {
//		WordNerdModel w = new WordNerdModel();
//		w.readWordsFile(w.WORDS_FILE_NAME);
//		for (int i = 0; i < wordsFromFile.length; i++) {
//			String temp = wordsFromFile[i];
//			System.out.println(temp);
//		}
//	}
	
}
