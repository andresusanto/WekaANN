/*
 *    MLP.java
 *    Copyright (C) 2015 by Andre Susanto, Adhika Sigit, Michael Alexander
 *    
 *    Implementation of Multi Layer Perceptron Classifier Algorithm
 */

package weka.classifiers.ann;

import weka.classifiers.Classifier;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.neural.NeuralNode;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Enumeration;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;


public class MLP extends Classifier implements OptionHandler, WeightedInstancesHandler, Randomizable {
  
  private static final long serialVersionUID = -5990607817048210779L;

  /**
   * Main method for testing this class.
   *
   * @param argv should contain command line options (see setOptions)
   */
  public static void main(String [] argv) {
    runClassifier(new MLP(), argv);
  }
  

  
  /** a ZeroR model in case no model can be built from the data 
   * or the network predicts all zeros for the classes */
  private Classifier m_ZeroR;

  /** Whether to use the default ZeroR model */
  private boolean m_useDefaultModel = false;
    
  /** The training instances. */
  private Instances m_instances;
  
  /** The current instance running through the network. */
  private Instance m_currentInstance;
  
  /** A flag to say that it's a numeric class. */
  private boolean m_numeric;

  /** The ranges for all the attributes. */
  private double[] m_attributeRanges;

  /** The base values for all the attributes. */
  private double[] m_attributeBases;

  /** The output units.(only feeds the errors, does no calcs) */
  private NeuralEnd[] m_outputs;

  /** The input units.(only feeds the inputs does no calcs) */
  private NeuralEnd[] m_inputs;

  /** All the nodes that actually comprise the logical neural net. */
  private NeuralConnection[] m_neuralNodes;

  /** The number of classes. */
  private int m_numClasses = 0;
  
  /** The number of attributes. */
  private int m_numAttributes = 0; //note the number doesn't include the class.
  
  /** The panel the nodes are displayed on. */
  private NodePanel m_nodePanel;
  
  /** The control panel. */
  private ControlPanel m_controlPanel;

  /** The next id number available for default naming. */
  private int m_nextId;
   
  /** A Vector list of the units currently selected. */
  private FastVector m_selected;

  /** A Vector list of the graphers. */
  private FastVector m_graphers;

  /** The number of epochs to train through. */
  private int m_numEpochs;

  /** a flag to state if the network should be running, or stopped. */
  private boolean m_stopIt;

  /** a flag to state that the network has in fact stopped. */
  private boolean m_stopped;

  /** a flag to state that the network should be accepted the way it is. */
  private boolean m_accepted;
  /** The window for the network. */
  private JFrame m_win;

  /** A flag to tell the build classifier to automatically build a neural net.
   */
  private boolean m_autoBuild;

  /** A flag to state that the gui for the network should be brought up.
      To allow interaction while training. */
  private boolean m_gui;

  /** An int to say how big the validation set should be. */
  private int m_valSize;

  /** The number to to use to quit on validation testing. */
  private int m_driftThreshold;

  /** The number used to seed the random number generator. */
  private int m_randomSeed;

  /** The actual random number generator. */
  private Random m_random;

  /** A flag to state that a nominal to binary filter should be used. */
  private boolean m_useNomToBin;
  
  /** The actual filter. */
  private NominalToBinary m_nominalToBinaryFilter;

  /** The string that defines the hidden layers */
  private String m_hiddenLayers;

  /** This flag states that the user wants the input values normalized. */
  private boolean m_normalizeAttributes;

  /** This flag states that the user wants the learning rate to decay. */
  private boolean m_decay;

  /** This is the learning rate for the network. */
  private double m_learningRate;

  /** This is the momentum for the network. */
  private double m_momentum;

  /** Shows the number of the epoch that the network just finished. */
  private int m_epoch;

  /** Shows the error of the epoch that the network just finished. */
  private double m_error;

  /** This flag states that the user wants the network to restart if it
   * is found to be generating infinity or NaN for the error value. This
   * would restart the network with the current options except that the
   * learning rate would be smaller than before, (perhaps half of its current
   * value). This option will not be available if the gui is chosen (if the
   * gui is open the user can fix the network themselves, it is an 
   * architectural minefield for the network to be reset with the gui open). */
  private boolean m_reset;

  /** This flag states that the user wants the class to be normalized while
   * processing in the network is done. (the final answer will be in the
   * original range regardless). This option will only be used when the class
   * is numeric. */
  private boolean m_normalizeClass;

  /**
   * this is a sigmoid unit. 
   */
  private SigmoidUnit m_sigmoidUnit;
  
  /**
   * This is a linear unit.
   */
  private LinearUnit m_linearUnit;
  
  /**
   * The constructor.
   */
  public MLP() {
    m_instances = null;
    m_currentInstance = null;
    m_controlPanel = null;
    m_nodePanel = null;
    m_epoch = 0;
    m_error = 0;
    
    
    m_outputs = new NeuralEnd[0];
    m_inputs = new NeuralEnd[0];
    m_numAttributes = 0;
    m_numClasses = 0;
    m_neuralNodes = new NeuralConnection[0];
    m_selected = new FastVector(4);
    m_graphers = new FastVector(2);
    m_nextId = 0;
    m_stopIt = true;
    m_stopped = true;
    m_accepted = false;
    m_numeric = false;
    m_random = null;
    m_nominalToBinaryFilter = new NominalToBinary();
    m_sigmoidUnit = new SigmoidUnit();
    m_linearUnit = new LinearUnit();
    //setting all the options to their defaults. To completely change these
    //defaults they will also need to be changed down the bottom in the 
    //setoptions function (the text info in the accompanying functions should 
    //also be changed to reflect the new defaults
    m_normalizeClass = true;
    m_normalizeAttributes = true;
    m_autoBuild = true;
    m_gui = false;
    m_useNomToBin = true;
    m_driftThreshold = 20;
    m_numEpochs = 500;
    m_valSize = 0;
    m_randomSeed = 0;
    m_hiddenLayers = "a";
    m_learningRate = .3;
    m_momentum = .2;
    m_reset = true;
    m_decay = false;
  }

  /**
   * @param d True if the learning rate should decay.
   */
  public void setDecay(boolean d) {
    m_decay = d;
  }
  
  /**
   * @return the flag for having the learning rate decay.
   */
  public boolean getDecay() {
    return m_decay;
  }

  /**
   * This sets the network up to be able to reset itself with the current 
   * settings and the learning rate at half of what it is currently. This
   * will only happen if the network creates NaN or infinite errors. Also this
   * will continue to happen until the network is trained properly. The 
   * learning rate will also get set back to it's original value at the end of
   * this. This can only be set to true if the GUI is not brought up.
   * @param r True if the network should restart with it's current options
   * and set the learning rate to half what it currently is.
   */
  public void setReset(boolean r) {
    if (m_gui) {
      r = false;
    }
    m_reset = r;
      
  }

  /**
   * @return The flag for reseting the network.
   */
  public boolean getReset() {
    return m_reset;
  }
  
  /**
   * @param c True if the class should be normalized (the class will only ever
   * be normalized if it is numeric). (Normalization puts the range between
   * -1 - 1).
   */
  public void setNormalizeNumericClass(boolean c) {
    m_normalizeClass = c;
  }
  
  /**
   * @return The flag for normalizing a numeric class.
   */
  public boolean getNormalizeNumericClass() {
    return m_normalizeClass;
  }

  /**
   * @param a True if the attributes should be normalized (even nominal
   * attributes will get normalized here) (range goes between -1 - 1).
   */
  public void setNormalizeAttributes(boolean a) {
    m_normalizeAttributes = a;
  }

  /**
   * @return The flag for normalizing attributes.
   */
  public boolean getNormalizeAttributes() {
    return m_normalizeAttributes;
  }

  /**
   * @param f True if a nominalToBinary filter should be used on the
   * data.
   */
  public void setNominalToBinaryFilter(boolean f) {
    m_useNomToBin = f;
  }

  /**
   * @return The flag for nominal to binary filter use.
   */
  public boolean getNominalToBinaryFilter() {
    return m_useNomToBin;
  }

  /**
   * This seeds the random number generator, that is used when a random
   * number is needed for the network.
   * @param l The seed.
   */
  public void setSeed(int l) {
    if (l >= 0) {
      m_randomSeed = l;
    }
  }
  
  /**
   * @return The seed for the random number generator.
   */
  public int getSeed() {
    return m_randomSeed;
  }

  /**
   * This sets the threshold to use for when validation testing is being done.
   * It works by ending testing once the error on the validation set has 
   * consecutively increased a certain number of times.
   * @param t The threshold to use for this.
   */
  public void setValidationThreshold(int t) {
    if (t > 0) {
      m_driftThreshold = t;
    }
  }

  /**
   * @return The threshold used for validation testing.
   */
  public int getValidationThreshold() {
    return m_driftThreshold;
  }
  
  /**
   * The learning rate can be set using this command.
   * NOTE That this is a static variable so it affect all networks that are
   * running.
   * Must be greater than 0 and no more than 1.
   * @param l The New learning rate. 
   */
  public void setLearningRate(double l) {
    if (l > 0 && l <= 1) {
      m_learningRate = l;
    
      if (m_controlPanel != null) {
	m_controlPanel.m_changeLearning.setText("" + l);
      }
    }
  }

  /**
   * @return The learning rate for the nodes.
   */
  public double getLearningRate() {
    return m_learningRate;
  }

  /**
   * The momentum can be set using this command.
   * THE same conditions apply to this as to the learning rate.
   * @param m The new Momentum.
   */
  public void setMomentum(double m) {
    if (m >= 0 && m <= 1) {
      m_momentum = m;
  
      if (m_controlPanel != null) {
	m_controlPanel.m_changeMomentum.setText("" + m);
      }
    }
  }
  
  /**
   * @return The momentum for the nodes.
   */
  public double getMomentum() {
    return m_momentum;
  }

  /**
   * This will set whether the network is automatically built
   * or if it is left up to the user. (there is nothing to stop a user
   * from altering an autobuilt network however). 
   * @param a True if the network should be auto built.
   */
  public void setAutoBuild(boolean a) {
    if (!m_gui) {
      a = true;
    }
    m_autoBuild = a;
  }

  /**
   * @return The auto build state.
   */
  public boolean getAutoBuild() {
    return m_autoBuild;
  }


  /**
   * This will set what the hidden layers are made up of when auto build is
   * enabled. Note to have no hidden units, just put a single 0, Any more
   * 0's will indicate that the string is badly formed and make it unaccepted.
   * Negative numbers, and floats will do the same. There are also some
   * wildcards. These are 'a' = (number of attributes + number of classes) / 2,
   * 'i' = number of attributes, 'o' = number of classes, and 't' = number of
   * attributes + number of classes.
   * @param h A string with a comma seperated list of numbers. Each number is 
   * the number of nodes to be on a hidden layer.
   */
  public void setHiddenLayers(String h) {
    String tmp = "";
    StringTokenizer tok = new StringTokenizer(h, ",");
    if (tok.countTokens() == 0) {
      return;
    }
    double dval;
    int val;
    String c;
    boolean first = true;
    while (tok.hasMoreTokens()) {
      c = tok.nextToken().trim();

      if (c.equals("a") || c.equals("i") || c.equals("o") || 
	       c.equals("t")) {
	tmp += c;
      }
      else {
	dval = Double.valueOf(c).doubleValue();
	val = (int)dval;
	
	if ((val == dval && (val != 0 || (tok.countTokens() == 0 && first)) && 
	     val >= 0)) {
	  tmp += val;
	}
	else {
	  return;
	}
      }
      
      first = false;
      if (tok.hasMoreTokens()) {
	tmp += ", ";
      }
    }
    m_hiddenLayers = tmp;
  }

  /**
   * @return A string representing the hidden layers, each number is the number
   * of nodes on a hidden layer.
   */
  public String getHiddenLayers() {
    return m_hiddenLayers;
  }

  /**
   * This will set whether A GUI is brought up to allow interaction by the user
   * with the neural network during training.
   * @param a True if gui should be created.
   */
  public void setGUI(boolean a) {
    m_gui = a;
    if (!a) {
      setAutoBuild(true);
      
    }
    else {
      setReset(false);
    }
  }

  /**
   * @return The true if should show gui.
   */
  public boolean getGUI() {
    return m_gui;
  }

  /**
   * This will set the size of the validation set.
   * @param a The size of the validation set, as a percentage of the whole.
   */
  public void setValidationSetSize(int a) {
    if (a < 0 || a > 99) {
      return;
    }
    m_valSize = a;
  }

  /**
   * @return The percentage size of the validation set.
   */
  public int getValidationSetSize() {
    return m_valSize;
  }

  
  
  
  /**
   * Set the number of training epochs to perform.
   * Must be greater than 0.
   * @param n The number of epochs to train through.
   */
  public void setTrainingTime(int n) {
    if (n > 0) {
      m_numEpochs = n;
    }
  }

  /**
   * @return The number of epochs to train through.
   */
  public int getTrainingTime() {
    return m_numEpochs;
  }
  

  /**
   * A function used to stop the code that called buildclassifier
   * from continuing on before the user has finished the decision tree.
   * @param tf True to stop the thread, False to release the thread that is
   * waiting there (if one).
   */
  public synchronized void blocker(boolean tf) {
    if (tf) {
      try {
	wait();
      } catch(InterruptedException e) {
      }
    }
    else {
      notifyAll();
    }
  }


  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }
  
  
  /**
   * Call this function to build and train a neural network for the training
   * data provided.
   * @param i The training data.
   * @throws Exception if can't build classification properly.
   */
  public void buildClassifier(Instances i) throws Exception {
		
  }

  /**
   * Call this function to predict the class of an instance once a 
   * classification model has been built with the buildClassifier call.
   * @param i The instance to classify.
   * @return A double array filled with the probabilities of each class type.
   * @throws Exception if can't classify instance.
   */
  public double[] distributionForInstance(Instance i) throws Exception {

  }
  


  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector(14);

    newVector.addElement(new Option(
	      "\tLearning Rate for the backpropagation algorithm.\n"
	      +"\t(Value should be between 0 - 1, Default = 0.3).",
	      "L", 1, "-L <learning rate>"));
    newVector.addElement(new Option(
	      "\tMomentum Rate for the backpropagation algorithm.\n"
	      +"\t(Value should be between 0 - 1, Default = 0.2).",
	      "M", 1, "-M <momentum>"));
    newVector.addElement(new Option(
	      "\tNumber of epochs to train through.\n"
	      +"\t(Default = 500).",
	      "N", 1,"-N <number of epochs>"));
    newVector.addElement(new Option(
	      "\tPercentage size of validation set to use to terminate\n"
	      + "\ttraining (if this is non zero it can pre-empt num of epochs.\n"
	      +"\t(Value should be between 0 - 100, Default = 0).",
	      "V", 1, "-V <percentage size of validation set>"));
    newVector.addElement(new Option(
	      "\tThe value used to seed the random number generator\n"
	      + "\t(Value should be >= 0 and and a long, Default = 0).",
	      "S", 1, "-S <seed>"));
    newVector.addElement(new Option(
	      "\tThe consequetive number of errors allowed for validation\n"
	      + "\ttesting before the netwrok terminates.\n"
	      + "\t(Value should be > 0, Default = 20).",
	      "E", 1, "-E <threshold for number of consequetive errors>"));
    newVector.addElement(new Option(
              "\tGUI will be opened.\n"
	      +"\t(Use this to bring up a GUI).",
	      "G", 0,"-G"));
    newVector.addElement(new Option(
              "\tAutocreation of the network connections will NOT be done.\n"
	      +"\t(This will be ignored if -G is NOT set)",
	      "A", 0,"-A"));
    newVector.addElement(new Option(
              "\tA NominalToBinary filter will NOT automatically be used.\n"
	      +"\t(Set this to not use a NominalToBinary filter).",
	      "B", 0,"-B"));
    newVector.addElement(new Option(
	      "\tThe hidden layers to be created for the network.\n"
	      + "\t(Value should be a list of comma separated Natural \n"
	      + "\tnumbers or the letters 'a' = (attribs + classes) / 2, \n"
	      + "\t'i' = attribs, 'o' = classes, 't' = attribs .+ classes)\n"
	      + "\tfor wildcard values, Default = a).",
	      "H", 1, "-H <comma seperated numbers for nodes on each layer>"));
    newVector.addElement(new Option(
              "\tNormalizing a numeric class will NOT be done.\n"
	      +"\t(Set this to not normalize the class if it's numeric).",
	      "C", 0,"-C"));
    newVector.addElement(new Option(
              "\tNormalizing the attributes will NOT be done.\n"
	      +"\t(Set this to not normalize the attributes).",
	      "I", 0,"-I"));
    newVector.addElement(new Option(
              "\tReseting the network will NOT be allowed.\n"
	      +"\t(Set this to not allow the network to reset).",
	      "R", 0,"-R"));
    newVector.addElement(new Option(
              "\tLearning rate decay will occur.\n"
	      +"\t(Set this to cause the learning rate to decay).",
	      "D", 0,"-D"));
    
    
    return newVector.elements();
  }

  
  // OPSI UNTUK KUSTOMISASI
  public void setOptions(String[] options) throws Exception {
    //the defaults can be found here!!!!
    String learningString = Utils.getOption('L', options);
    if (learningString.length() != 0) {
      setLearningRate((new Double(learningString)).doubleValue());
    } else {
      setLearningRate(0.3);
    }
    String momentumString = Utils.getOption('M', options);
    if (momentumString.length() != 0) {
      setMomentum((new Double(momentumString)).doubleValue());
    } else {
      setMomentum(0.2);
    }
    String epochsString = Utils.getOption('N', options);
    if (epochsString.length() != 0) {
      setTrainingTime(Integer.parseInt(epochsString));
    } else {
      setTrainingTime(500);
    }
    String valSizeString = Utils.getOption('V', options);
    if (valSizeString.length() != 0) {
      setValidationSetSize(Integer.parseInt(valSizeString));
    } else {
      setValidationSetSize(0);
    }
    String seedString = Utils.getOption('S', options);
    if (seedString.length() != 0) {
      setSeed(Integer.parseInt(seedString));
    } else {
      setSeed(0);
    }
    String thresholdString = Utils.getOption('E', options);
    if (thresholdString.length() != 0) {
      setValidationThreshold(Integer.parseInt(thresholdString));
    } else {
      setValidationThreshold(20);
    }
    String hiddenLayers = Utils.getOption('H', options);
    if (hiddenLayers.length() != 0) {
      setHiddenLayers(hiddenLayers);
    } else {
      setHiddenLayers("a");
    }
    if (Utils.getFlag('G', options)) {
      setGUI(true);
    } else {
      setGUI(false);
    } //small note. since the gui is the only option that can change the other
    //options this should be set first to allow the other options to set 
    //properly
    if (Utils.getFlag('A', options)) {
      setAutoBuild(false);
    } else {
      setAutoBuild(true);
    }
    if (Utils.getFlag('B', options)) {
      setNominalToBinaryFilter(false);
    } else {
      setNominalToBinaryFilter(true);
    }
    if (Utils.getFlag('C', options)) {
      setNormalizeNumericClass(false);
    } else {
      setNormalizeNumericClass(true);
    }
    if (Utils.getFlag('I', options)) {
      setNormalizeAttributes(false);
    } else {
      setNormalizeAttributes(true);
    }
    if (Utils.getFlag('R', options)) {
      setReset(false);
    } else {
      setReset(true);
    }
    if (Utils.getFlag('D', options)) {
      setDecay(true);
    } else {
      setDecay(false);
    }
    
    Utils.checkForRemainingOptions(options);
  }
  
  /**
   * Gets the current settings of NeuralNet.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [21];
    int current = 0;
    options[current++] = "-L"; options[current++] = "" + getLearningRate(); 
    options[current++] = "-M"; options[current++] = "" + getMomentum();
    options[current++] = "-N"; options[current++] = "" + getTrainingTime(); 
    options[current++] = "-V"; options[current++] = "" +getValidationSetSize();
    options[current++] = "-S"; options[current++] = "" + getSeed();
    options[current++] = "-E"; options[current++] =""+getValidationThreshold();
    options[current++] = "-H"; options[current++] = getHiddenLayers();
    if (getGUI()) {
      options[current++] = "-G";
    }
    if (!getAutoBuild()) {
      options[current++] = "-A";
    }
    if (!getNominalToBinaryFilter()) {
      options[current++] = "-B";
    }
    if (!getNormalizeNumericClass()) {
      options[current++] = "-C";
    }
    if (!getNormalizeAttributes()) {
      options[current++] = "-I";
    }
    if (!getReset()) {
      options[current++] = "-R";
    }
    if (getDecay()) {
      options[current++] = "-D";
    }

    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }
  
  
  /**
   * @return string describing the model.
   */
  public String toString() {
    return "OUTPUT MODEL";
  }

  
  /**
   * This will return a string describing the classifier.
   * @return The string.
   */
  public String globalInfo() {
    return "Multi Layer Perceptron";
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10073 $");
  }
}
