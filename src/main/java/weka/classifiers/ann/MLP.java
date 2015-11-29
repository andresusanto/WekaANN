/*
 *    MLP.java
 *    Copyright (C) 2015 by Andre Susanto, Adhika Sigit, Michael Alexander
 *    
 *    Implementation of MULTI LAYER PERCEPTRON Classifier Algorithm
 */


package weka.classifiers.ann;

import weka.classifiers.Classifier;
import weka.classifiers.ann.engine.Links;
import weka.classifiers.ann.engine.Node;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;
import java.util.Enumeration;

public class MLP extends Classifier implements OptionHandler, WeightedInstancesHandler {
    ////// DATA YANG BERKAITAN DENGAN MODEL
    private ArrayList<ArrayList<Node>> hidden;
    private ArrayList<Node> input;
    private ArrayList<Node> output;
    private HashMap<Node,Integer> inputIndex, outputIndex;
    private HashMap<Integer,HashMap<Node,Integer>> hiddenIndex;

    private double momentum = 0.0;
    private float learningrate = 0.1f;
    private int maxIteration = 30;
    private int hiddenPerceptrons = 1;
    private int hiddenLayers = 1;
    private int outPerceptron = 1;
    private boolean useFilter = false;
    private int initOption = 0;
    private double initValue = 0.0;
    private double deltaMSE = 0.0;
    ///////////////////////////////////////
    private NominalToBinary nominalToBinary = new NominalToBinary();
    private Normalize normalize = new Normalize();
    //////////////////////////////////////


    // untuk serialisasi
    private static final long serialVersionUID = -5990607817048210779L;

    private void initWeight(){
        for(Node i : this.input){
            for(Node h : this.hidden.get(0)){
                if (initOption == 0)
                    i.forbind(h, Math.random()*(Math.random() > 0.5 ? 1 : -1));
                else
                    i.forbind(h, initValue);
            }
        }
        for(int i = 1; i < this.hidden.size(); i++){
            for(Node h : this.hidden.get(i-1)){
                for(Node hto : this.hidden.get(i)){
                    if (initOption == 0)
                        h.forbind(hto, Math.random()*(Math.random() > 0.5 ? 1 : -1));
                    else
                        h.forbind(hto, initValue);
                }
            }
        }
        for(Node h : this.hidden.get(this.hidden.size()-1)){
            for(Node o : this.output){
                if (initOption == 0)
                    h.forbind(o, Math.random()*(Math.random() > 0.5 ? 1 : -1));
                else
                    h.forbind(o, initValue);
            }
        }
    }

    private double[] nominalize(double in){
        double out[] = new double[outPerceptron];

        if (outPerceptron == 1){
            out[0] = in;
        }else{
            for (int i = 0; i < outPerceptron; i++){
                out[i] = 0.01;
            }
            out[(int)in] = 0.99;
        }
        return out;
    }

    private double calculateError(Instances instances) throws Exception{
        double tmp_error = 0;
        int sumInstances = instances.numInstances();

        for (int i = 0; i < sumInstances ; i++){
            tmp_error += Math.pow( instances.instance(i).classValue() - classifyInstance(instances.instance(i)) , 2);
        }
        return tmp_error / 2;
    }

    private double[] genInput(Instance instance){
        int it = 0;
        double[] input = new double[instance.numAttributes() - 1];
        for (int i = 0; i < instance.numAttributes(); i++){
            if (i != instance.classIndex()){
                input[it] = instance.value(i);
                it++;
            }
        }
        return input;
    }

    public void buildClassifier(Instances _instances) throws Exception {
        Instances instances = _instances;
        getCapabilities().testWithFail(_instances);

        /* FILTERING OPTIONS  */
        if (useFilter) {
            nominalToBinary.setInputFormat(instances);
            instances = Filter.useFilter(instances, nominalToBinary);
            normalize.setInputFormat(instances);
            instances = Filter.useFilter(instances, normalize);
        }

        int i = 0; int it = 0;
        int sumInstances = instances.numInstances();
        int sumAttributes = instances.numAttributes();
        int iterateTo = maxIteration * sumInstances;
        if (instances.classAttribute().isNominal())
            outPerceptron = instances.classAttribute().numValues();

        prepare(sumAttributes - 1, hiddenPerceptrons, outPerceptron, hiddenLayers);

        double curError = 1;

        while (it < iterateTo && curError > deltaMSE) {
            double input[] = genInput(instances.instance(i));
            double target[] = nominalize(instances.instance(i).classValue());

            train(input, target);

            curError = calculateError(instances);
            i = (++i) % sumInstances;
            it++;
        }
    }


    public double classifyInstance(Instance _instance) throws Exception{
        Instance instance = _instance;

        /* FILTERING OPTIONS  */
        if (useFilter){
            nominalToBinary.input(instance);
            instance = nominalToBinary.output();
            normalize.input(instance);
            instance = normalize.output();
        }

        double input[] = genInput(instance);
        double result[] = classify(input);

        if (outPerceptron > 1){
            int maxIndex = 0; double maxVal = 0;
            for (int l = 0; l < outPerceptron ; l++){
                if (l == 0 || maxVal < result[l]){
                    maxIndex = l;
                    maxVal = result[l];
                }
            }

            return  maxIndex;
        }else {
            return result[0];
        }

    }

    private double[] classify(double[] input) {
        for(int i = 0; i < input.length; i++){
            this.input.get(i).input(input[i]);
        }
        double[] r = new double[this.output.size()];
        for(int i = 0; i < r.length; i++){
            r[i] = this.output.get(i).getSenesteOutput();
        }
        return r;
    }

    private double[] map(double[] input){
        for(int i = 0; i < input.length; i++){
            this.input.get(i).input(input[i]);
        }
        double[] retur = new double[this.output.size()];
        for(int i = 0; i < retur.length; i++){
            retur[i] = this.output.get(i).getSenesteOutput();
        }
        return retur;
    }

    private void train(double[] input, double[] target){
        for(int i = 0; i < input.length; i++){
            this.input.get(i).input(input[i]);
        }
        backpropagate(target);
    }

    private void backpropagate(double[] exp){
        double[] error = new double[this.output.size()];

        int c = 0;
        for(Node o : this.output){
            error[c] = o.getSenesteOutput()*(1.0-o.getSenesteOutput())*(exp[this.outputIndex.get(o)]-o.getSenesteOutput());
            c++;
        }
        for(Node h : this.hidden.get(this.hidden.size()-1)){
            for(Links s : h.getForbundetTil()){
                double v = s.getV();
                double p = s.getP();
                s.setV(v + this.learningrate * h.getSenesteOutput() * error[this.outputIndex.get(s.getTil())] + momentum * p);
                s.setP(this.learningrate * h.getSenesteOutput() * error[this.outputIndex.get(s.getTil())] + momentum * p);
            }
        }
        double[] oerror = error.clone();
        error = new double[this.hidden.get(0).size()];

        for(int i = this.hidden.size()-1; i > 0; i--){
            c = 0;
            for(Node h : this.hidden.get(i)){
                double p = h.getSenesteOutput()*(1-h.getSenesteOutput());
                double k = 0;
                for(Links s : h.getForbundetTil()){
                    if(i == this.hidden.size()-1){
                        k = k+oerror[this.outputIndex.get(s.getTil())]*s.getV();
                    }
                    else{
                        k = k+error[this.hiddenIndex.get(i+1).get(s.getTil())]*s.getV();
                    }
                }
                error[c] = p*k;
                c++;
            }
            for(Node h : this.hidden.get(i-1)){
                for(Links s : h.getForbundetTil()){
                    double v = s.getV();
                    double p = s.getP();

                    int index = this.hiddenIndex.get(i).get(s.getTil());
                    s.setV(v + this.learningrate * error[index] * h.getSenesteInput() + momentum * p);
                    s.setP(this.learningrate * error[index] * h.getSenesteInput() + momentum * p);
                }
            }
        }

        c = 0;
        double[] t = error.clone();
        for(Node h : this.hidden.get(0)){
            double p = h.getSenesteOutput()*(1.0-h.getSenesteOutput());
            double k = 0;
            for(Links s : h.getForbundetTil()){
                if(this.hidden.size() == 1){
                    k = k+s.getV()*oerror[this.outputIndex.get(s.getTil())];
                }
                else{
                    k = k+s.getV()*error[this.hiddenIndex.get(1).get(s.getTil())];
                }
            }
            t[c] = k*p;
            c++;
        }
        for(Node i : this.input){
            for(Links s : i.getForbundetTil()){
                double v = s.getV();
                double p = s.getP();
                s.setV(v + this.learningrate * t[this.hiddenIndex.get(0).get(s.getTil())] * i.getSenesteInput() + momentum * p);
                s.setP(this.learningrate * t[this.hiddenIndex.get(0).get(s.getTil())] * i.getSenesteInput() + momentum * p);
            }
        }
    }

    private void prepare(int input, int hidden, int output, int numberOfHiddenLayers){
        this.hiddenIndex = new HashMap<Integer,HashMap<Node,Integer>>();
        this.inputIndex = new HashMap<Node,Integer>();
        this.outputIndex = new HashMap<Node,Integer>();

        this.hidden = new ArrayList<ArrayList<Node>>();
        this.input = new ArrayList<Node>();
        this.output = new ArrayList<Node>();

        // Tambah Input Layers
        for(int i = 1; i <= input; i++){
            this.input.add(new Node(false));
        }
        for(Node i : this.input){
            this.inputIndex.put(i, this.input.indexOf(i));
        }

        // Tambah Hidden Layers
        for(int i = 1; i <= numberOfHiddenLayers; i++){
            ArrayList<Node> a = new ArrayList<Node>();
            for(int j = 1; j <= hidden; j++){
                a.add(new Node(true));
            }
            this.hidden.add(a);
        }
        for(ArrayList<Node> a : this.hidden){
            HashMap<Node,Integer> put = new HashMap<Node,Integer>();
            for(Node h : a){
                put.put(h, a.indexOf(h));
            }
            this.hiddenIndex.put(this.hidden.indexOf(a), put);
        }

        // Tambah Output Layers
        for(int i = 1; i <= output; i++){
            this.output.add(new Node(true));
        }
        for(Node o : this.output){
            this.outputIndex.put(o, this.output.indexOf(o));
        }

        initWeight();
    }


    public Enumeration listOptions() {

        Vector newVector = new Vector(5);

        newVector.addElement(new Option(
                "\tNumber of Hidden Perceptrons in Hidden Layer.\n"
                        +"\t(Value should be >= 1.",
                "H", 1, "-H <activation function>"));

        newVector.addElement(new Option(
                "\tMaximum Iteration that is used by PTR.\n"
                        +"\t(Default 30).",
                "M", 1, "-M <max iteration>"));

        newVector.addElement(new Option(
                "\tLearning Rate.\n"
                        +"\t(Default = 0.1).",
                "L", 1,"-L <learning rate>"));

        newVector.addElement(new Option(
                "\tHidden Layers.\n"
                        +"\t(Default = 1).",
                "N", 1,"-N <hidden layers>"));

        newVector.addElement(new Option(
                "\tUse Filters.\n"
                        +"\t(Default = 0, input 1 to use filter).",
                "F", 1,"-F <use filter>"));


        return newVector.elements();
    }

    public String maxIterationTipText() {
        return "Maximum Iteration that is used by PTR. Default = 30";
    }

    public String activationFunctionTipText() {
        return "Activation Function that is used by PTR. (Value should be between 0 - 2, 0 = SIGN, 1 = STEP, 2 = SIGMOID).";
    }

    public String thresholdTipText() {
        return "Learning rate that is used by the algorithm. Default = 0";
    }

    public String learningRateTipText() {
        return "Learning rate that is used by the algorithm. Default = 0.1";
    }

    public String hiddenLayersTipText() {
        return "Numbers of Hidden Layers. Default = 1";
    }

    public String useFilterTipText() {
        return "Use filter. Default = 0, input 1 to use filter";
    }

    public String initWeightTipText() {
        return "Initial weight. Input `a` to Random, input number to init with number";
    }

    public String momentumTipText(){
        return "Momentum that is used to update weight (to avoid local maxima). Enter numeric value (0.0 - 1.0)";
    }

    public String deltaMseTipText(){
        return "Minimum Square Error to continue training (Enter value > 0)";
    }

    public void setDeltaMse(double a){
        deltaMSE = a;
    }

    public  double getDeltaMse(){
        return  deltaMSE;
    }

    public void setInitWeight(String a){
        if (a.equals("a")){
            initOption = 0;
        }else{
            initOption = 1;
            initValue = new Double(a).doubleValue();
        }
    }

    public String getInitWeight(){
        if (initOption == 0) return "a";
        return "" + initValue;
    }

    public void setMomentum(double a){
        momentum = a;
    }

    public double getMomentum(){
        return momentum;
    }

    public void setUseFilter(boolean a) {
        useFilter = a;
    }

    public boolean getUseFilter() {
        return useFilter;
    }

    public void setHiddenLayers(int a) {
        hiddenLayers = a;
    }

    public int getHiddenLayers() {
        return hiddenLayers;
    }

    public void setLearningRate(float a) {
        learningrate = a;
    }

    public float getLearningRate() {
        return learningrate;
    }

    public void setHiddenPerceptrons(int a) {
        hiddenPerceptrons = a;
    }

    public int getHiddenPerceptrons() {
        return hiddenPerceptrons;
    }


    public void setMaxIteration(int a) {
        maxIteration = a;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    // OPSI UNTUK KUSTOMISASI
    public void setOptions(String[] options) throws Exception {
        String learningString = Utils.getOption('L', options);
        if (learningString.length() != 0) {
            learningrate = new Float(learningString).floatValue();
        } else {
            learningrate = 0.1f;
        }

        String activationString = Utils.getOption('H', options);
        if (activationString.length() != 0) {
            hiddenPerceptrons = new Integer(activationString).intValue();
        } else {
            hiddenPerceptrons = 1;
        }

        String maxiterString = Utils.getOption('M', options);
        if (maxiterString.length() != 0) {
            maxIteration = new Integer(maxiterString).intValue();
        } else {
            maxIteration = 30;
        }

        String filterString = Utils.getOption('F', options);
        if (filterString.length() != 0) {
            useFilter = new Integer(filterString).intValue() == 1;
        } else {
            useFilter = false;
        }

        String hiddenString = Utils.getOption('N', options);
        if (hiddenString.length() != 0) {
            hiddenLayers = new Integer(hiddenString).intValue();
        } else {
            hiddenLayers = 1;
        }


        Utils.checkForRemainingOptions(options);
    }

    public int booleanint(boolean intv){
        if (intv) return 1;
        return 0;
    }

    public String [] getOptions() {
        String [] options = new String [10];
        int current = 0;
        options[current++] = "-L"; options[current++] = "" + learningrate;
        options[current++] = "-H"; options[current++] = "" + hiddenPerceptrons;
        options[current++] = "-M"; options[current++] = "" + maxIteration;
        options[current++] = "-N"; options[current++] = "" + hiddenLayers;
        options[current++] = "-F"; options[current++] = "" + booleanint(useFilter);

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }


    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        //result.enable(Capability.DATE_ATTRIBUTES);
        //result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        //result.enable(Capability.DATE_CLASS);
        //result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }

    public static void main(String [] argv) {
        runClassifier(new MLP(), argv);
    }



    public String toString() {
        return "OUTPUT MODEL";
    }


    public MLP() { }

    // Mengembalikan informasi mengenai classifier ini

    public String globalInfo() {
        return "Perceptron Training Rule";
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1 $");
    }
}
