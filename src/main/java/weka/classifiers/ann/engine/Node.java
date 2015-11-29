package weka.classifiers.ann.engine;


import java.io.Serializable;
import java.util.ArrayList;

public class Node implements Serializable {
	private boolean act;
	private int antalTriggered = 0, antallinkstil = 0;
	private double senesteinput = 0, senesteoutput = 0, sum;
	private ArrayList<Links> forbundetTil;
	public Node(boolean act){
		this.act = act;
		this.forbundetTil = new ArrayList<Links>();
	}
	
	public void forbind(Node e, double v){
		Links n = new Links(e, v);
		this.forbundetTil.add(n);
		e.LinksTil();
	}
	
	public double getSenesteInput(){
		return this.senesteinput;
	}
	
	public double getSenesteOutput(){
		return this.senesteoutput;
	}
	
	public void input(double input){
		this.antalTriggered++;
		this.sum = sum+input;
		if(this.antalTriggered >= this.antallinkstil){
			this.senesteinput = sum;
			test();
		}
	}
	
	public void test(){
		for(Links n : this.forbundetTil){
			if(this.act){
				n.getTil().input(activation(this.sum)*n.getV());
			}
			else{
				n.getTil().input(this.sum*n.getV());
			}
		}
		if(this.act){
			this.senesteoutput = activation(this.sum);
		}
		else{
			this.senesteoutput = this.sum;
		}
		this.sum = 0.0;
		this.antalTriggered = 0;
	}
	
	public void LinksTil(){
		this.antallinkstil++;
	}
	
	public ArrayList<Links> getForbundetTil(){
		return this.forbundetTil;
	}
	
	public String toString(){
		String retur = this.hashCode()+" med "+this.forbundetTil.size()+" forbindelser.";
		return retur;
	}

    private static double activation(double x){
        return 1.0/(1+Math.pow(Math.E, -x));
    }
}