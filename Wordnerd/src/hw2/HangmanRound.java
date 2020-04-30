package hw2;

import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;

public class HangmanRound extends GameRound{
	
	private IntegerProperty hitcount = new SimpleIntegerProperty();
	private IntegerProperty misscount = new SimpleIntegerProperty();
	
	//Hitcount
	public int getHitCount() {return this.hitcount.get();}
	public void setHitCount(int hc) {this.hitcount.set(hc);}
	public IntegerProperty hitcountProperty() {return this.hitcount;}
	
	//Misscount
	public int getMissCount() {return misscount.get();}
	public void setMissCount(int mc) {this.misscount.set(mc);}
	public IntegerProperty misscountProperty() {return this.misscount;}

}
