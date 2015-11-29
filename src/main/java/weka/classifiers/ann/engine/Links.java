package weka.classifiers.ann.engine;


import java.io.Serializable;

public class Links implements Serializable {
	private Node til;
	private double v;
	public Links(Node til, double v){
		this.til = til;
		this.v = v;
	}
	
	public double getV(){
		return this.v;
	}
	
	public void setV(double v){
		this.v = v;
	}
	
	public Node getTil(){
		return this.til;
	}
	
	public String toString(){
		return v+"";
	}
}