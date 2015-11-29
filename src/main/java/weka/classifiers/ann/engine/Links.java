package weka.classifiers.ann.engine;


import java.io.Serializable;

public class Links implements Serializable {
	private Node til;
	private double v, p;

	public Links(Node til, double v){
		this.til = til;
		this.v = v;
        p = 0;
	}

    public void setP(double p){
        this.p = p;
    }

    public double getP(){
        return this.p;
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