package trial.aparapi;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.exp;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public class AparapiTrial {
    public static final double ABSOLUTE_FLOAT_ERROR_TOLERANCE = 2e-4;
    public static final double RELATIVE_FLOAT_ERROR_TOLERANCE = 5e-8;

    private static class MyKernelDouble extends Kernel {
    	private double[] vector1;
    	private double[] vector2;
    	private double[] outputVector;
    	private boolean[] selectedRepeating;
    	private int[] sel;
    	
    	public MyKernelDouble() {
			super();
			setExplicit(true);
		}

    	public MyKernelDouble setVector1(double[] val) {
    		this.vector1 = val;
    		return this;
    	}

    	public MyKernelDouble setVector2(double[] val) {
    		this.vector2 = val;
    		return this;
    	}

    	public MyKernelDouble setOutputVector(double[] val) {
    		this.outputVector = val;
    		return this;
    	}

		public MyKernelDouble setSelectedRepeating(boolean[] val) {
			this.selectedRepeating = val;
			return this;
		}
		
		public MyKernelDouble setSel(int[] val) {
			this.sel = val;
			return this;
		}
		
		@Override
    	public void run() {
    		int j = getGlobalId();
    		if(selectedRepeating[1]) {
    			if(selectedRepeating[0]) {
    				int i = sel[j];
          			outputVector[i] = vector1[0] + vector2[i];
    			}
    			else {
    				outputVector[j] = vector1[0] + vector2[j];
    			}
    		}
    		else if(selectedRepeating[2]) {
    			if(selectedRepeating[0]) {
		            int i = sel[j];
		            outputVector[i] = vector1[i] + vector2[0];
    			}
    			else {
    				outputVector[j] = vector1[j] + vector2[0];
    			}
    		}
    		else {
    			if(selectedRepeating[0]) {
		            int i = sel[j];
		            outputVector[i] = vector1[i] + vector2[i];
    			}
    			else {
    				outputVector[j] = vector1[j] +  vector2[j];
    			}
    		}
    	}
    } // end class MyKernel
    
    private static void executeOnDevice(MyKernelDouble kernel, double[] vector1, double[] vector2, double[] outputVector, boolean[] selectedRepeating, int[] sel) {
    	System.out.println("Execution mode: " + kernel.getExecutionMode());
        long t1_g = System.currentTimeMillis();
        // send parameters and execute, copy the OpenCL-hosted array back to RAM
        kernel.setVector1(vector1).setVector2(vector2).setOutputVector(outputVector).setSelectedRepeating(selectedRepeating).setSel(sel);
        kernel.put(vector1).put(vector2).put(outputVector).put(selectedRepeating).put(sel);
        int kernelRange = vector1.length;
        if(selectedRepeating[1])
        	kernelRange = vector2.length;
        kernel.execute(kernelRange).get(outputVector);
		long t2_g = System.currentTimeMillis();
		System.out.println("Device " + kernel.getExecutionMode() + " time diff: " + (t2_g-t1_g) + " ms");
    }

    private static class MyKernel extends Kernel {
    	private float[] a;
    	private float[] b;
    	private float[] output;
    	private boolean[] selected;
    	
    	public MyKernel() {
			super();
			setExplicit(true);
		}

    	public MyKernel setA(float[] a) {
    		this.a = a;
    		return this;
    	}

    	public MyKernel setB(float[] b) {
    		this.b = b;
    		return this;
    	}

    	public MyKernel setOutput(float[] output) {
    		this.output = output;
    		return this;
    	}
    	
    	public MyKernel setSelected(boolean[] selected) {
    		this.selected = selected;
    		return this;
    	}

		@Override
    	public void run() {
    		int i = getGlobalId();
    		if(selected[0])
    			output[i] = sin(exp(cos(sin(a[i]) * sin(b[i]) + 1)));
    		else
    			output[i] = cos(exp(cos(sin(a[i]) * sin(b[i]) + 1)));
    		//float DISCOUNT = a[i];
    		//float QUANTITY = b[i];
    		//output[i] = exp(sin(cos(sin(1.0f+exp(cos(sin(sin(sqrt(8.0f+exp(1.0f+sin(pow(QUANTITY, 4.5f) - exp(DISCOUNT))))+2.0f)*exp(1.0f+QUANTITY)*cos(1.0f+exp(DISCOUNT))+2.0f))+5.0f)))))*sin(cos(cos(exp(1.0f+exp(sin(sin(sin(sqrt(8.0f+exp(1.0f+cos(pow(QUANTITY, sqrt(2.0f)) - sin(DISCOUNT))))+2.0f)*exp(2.0f+QUANTITY)*cos(3+exp(DISCOUNT*2.0f))+2.0f))+5.0f)))));
    	}
    }
    
    public static class ReducerKernel extends Kernel {
    	private float[] a;
    	private int[] iterationNo;
    	private float[] output;
    	
    	public ReducerKernel() {
			super();
			setExplicit(true);
		}

    	public ReducerKernel setA(float[] val) {
    		this.a = val;
    		return this;
    	}

    	public ReducerKernel setIterationNo(int[] val) {
    		this.iterationNo = val;
    		return this;
    	}

    	public ReducerKernel setOutput(float[] val) {
    		this.output = val;
    		return this;
    	}
    	
		@Override
    	public void run() {
			int i = getGlobalId(), iter = iterationNo[0];
			int localBatchSize = iterationNo[2];
    		int iterationMultiplier = iterationNo[3]; // localBatchSize ^ iterationNo[0]
    		int myBaseIndex = i*localBatchSize;
    		int myIndex = myBaseIndex*iterationMultiplier;
    		if(myIndex < iterationNo[1]) {
	    		for(int cnt = 1;cnt < localBatchSize;cnt++) {
	    			int nextIndex = (myBaseIndex+cnt)*iterationMultiplier;
	    			if(nextIndex < iterationNo[1])
	    				a[myIndex] += a[nextIndex];
	    		}
    		}
    		if(iter == iterationNo[4])
    			output[0] = a[0];
    	}
    }

    private static void executeOnDevice() {
        ReducerKernel kernel = new ReducerKernel();
        float[] vector = new float[1 << 20];
        for(int i = 0;i < vector.length;i++) {
        	vector[i] = (float) (i+1);
        }
    	//kernel.setExecutionMode(EXECUTION_MODE.JTP);
    	System.out.println("Execution mode: " + kernel.getExecutionMode());
    	long t1 = System.currentTimeMillis();
    	float sum = 0.0f;
    	for(int i = 0;i < vector.length;i++) {
    		sum += vector[i];
    	}
    	long t2 = System.currentTimeMillis();
    	System.out.println("Raw computed sum: " + ((long) sum) + ", time " + (t2-t1) + "ms");
    	// compute a dummy sum to compile the kernel
    	float[] output = new float[] { 0.0f };
    	{
    		float[] a = new float[] { 1.0f };
    		int[] iterNo = new int[] { 0, a.length, 2, 1, 1 };
    		kernel.setA(a).setOutput(output).setIterationNo(iterNo).put(a).put(output).put(iterNo).execute(1);
    	}
        long t1_g = System.currentTimeMillis();
        // send parameters and execute, copy the OpenCL-hosted array back to RAM
        int localBatchSize = 64;
        double logLength = Math.log(vector.length), logBatchSize = Math.log((double) localBatchSize); 
        int maxIters = (int) (logLength/logBatchSize);
        if(Math.abs(logLength - maxIters*logBatchSize) > 0.0001)
        	maxIters++;
        
        kernel.setA(vector).setOutput(output).put(vector).put(output);
        int[] iterationNo = new int[] { 0, vector.length, localBatchSize, 1, maxIters-1 };
        for(int iter = 0;iter < maxIters;iter++) {
        	int kernelRange = 0, iterationMultiplier = 0;
        	if(localBatchSize == 2) { // special optimization for batch size == 2. division by power of 2 is changed to right shift by power bits
        		iterationMultiplier = 1 << iter; // 2^iter
		        kernelRange = vector.length >> (iter+1); // vector.length/2^(iter+1)
        	}
        	else {
        		// compute localBatchSize ^ (iter+1)
        		int localBatchSizeIterFactor = localBatchSize;
        		for(int i = 1;i <= iter;i++) {
        			localBatchSizeIterFactor *= localBatchSize;
        		}
        		kernelRange = vector.length/localBatchSizeIterFactor;
        		if(kernelRange == 0)
        			kernelRange = 1;
        		iterationMultiplier = localBatchSizeIterFactor/localBatchSize; // localBatchSize ^ iter
        	}
        	iterationNo[0] = iter;
        	iterationNo[3] = iterationMultiplier;
	        kernel.setIterationNo(iterationNo).put(iterationNo).execute(kernelRange);
        }
        kernel.get(output);
		long t2_g = System.currentTimeMillis();
		System.out.println("Device " + kernel.getExecutionMode() + " time diff: " + (t2_g-t1_g) + " ms");
		System.out.println("Sum: " + ((long) output[0]));
    }

	private static class MyRunnable implements Runnable {
		private int start;
		private int end;
    	private float[] a;
    	private float[] b;
		public float[] expectedResults;
		private boolean selected;
		public int chunkSize;

		public MyRunnable(float[] a, float[] b, boolean selected, int start, int end) {
			this.start = start;
			this.end = end;
			this.chunkSize = end - start;
			this.a = a;
			this.b = b;
			this.selected = selected;
			int dataSize = end - start + 1;
			expectedResults = new float[dataSize];
		}
		
		public void run() {
			if(selected) {
				for (int i = start; i < end; i++) {
					expectedResults[i-start] = (float) sin(exp(cos(sin(a[i]) * sin(b[i]) + 1)));
				}
			}
			else {
				for (int i = start; i < end; i++) {
					expectedResults[i-start] = (float) cos(exp(cos(sin(a[i]) * sin(b[i]) + 1)));
				}
			}
		}
	}

	static {
		try {
		//System.setOut(new PrintStream(new FileOutputStream("c:/dev/aparapi/aparapi.out")));
		//System.setErr(new PrintStream(new FileOutputStream("c:/dev/aparapi/aparapi.err")));
		}
		catch(Exception ex) { }
	}
    public static void main(String[] args) {
        try {
			
            int dataSize = 100000;
			String dataSizeStr = System.getProperty("dataSize", Integer.toString(dataSize));
			if(args != null && args.length > 0 && !args[0].isEmpty())
				dataSizeStr = args[0];

			try {
				dataSize = Integer.parseInt(dataSizeStr);
			} catch(Exception ex) {}

            // Allocate OpenCL-hosted memory for inputs and output, 
            // with inputs initialized as copies of the NIO buffers
			final float[] a = new float[dataSize];
			final float[] b = new float[dataSize];
			final float[] output = new float[dataSize];
			final double[] vector1 = new double[dataSize];
			final double[] vector2 = new double[dataSize];
			final double[] outputVector = new double[dataSize];
			int[] sel = new int[dataSize];
			//final MyStruct[] s = new MyStruct[dataSize];
            for (int i = 0; i < dataSize; i++) {
                float value = (float)i;
                a[i] = value;
                b[i] = value;
                vector1[i] = (double) value;
                vector2[i] = (double) value;
                //s[i] = new MyStruct(value, value);
            }
            executeOnDevice();
            
            MyKernel kernel = new MyKernel();
            MyKernelDouble kernelDouble = new MyKernelDouble();
            boolean selected = true;
            executeOnDevice(kernel, a, b, output, new boolean[] { selected });
            executeOnDevice(kernelDouble, vector1, vector2, outputVector, new boolean[] {false, false, false}, sel);
			MyRunnable[] tasks = executeOnJavaThreads(a, b, selected);
			computeDifference(output, tasks);

            for (int i = 0; i < dataSize; i++) {
                float value = (float)i;
                a[i] = 2*value+1;
                b[i] = value+2;
                //s[i] = new MyStruct(value, value);
            }
            selected = false;
            executeOnDevice(kernel, a, b, output, new boolean[] { selected });
			tasks = executeOnJavaThreads(a, b, selected);
			computeDifference(output, tasks);
            /*
            MyStructuredKernel sKernel = new MyStructuredKernel(s);
            sKernel.execute(s.length);
            System.out.println("Structure: First time execution time: " + sKernel.getExecutionTime() + "ms");
            sKernel.execute(s.length);
            System.out.println("Structure: Second time execution time: " + sKernel.getExecutionTime() + "ms");
            */
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void computeDifference(float[] output, MyRunnable[] tasks) {
        double totalAbsoluteError = 0, totalRelativeError = 0;
		for(int i = 0;i < output.length;i++) {
			float expected = tasks[i/tasks[0].chunkSize].expectedResults[i % tasks[0].chunkSize];
            float result = output[i];
            double d = result - expected;
            if (expected != 0) {
                totalRelativeError += d / expected;
            }

            totalAbsoluteError += d < 0 ? -d : d;
        }
        double avgAbsoluteError = totalAbsoluteError / output.length;
        double avgRelativeError = totalRelativeError / output.length;
        System.out.println("Average absolute error = " + avgAbsoluteError);
        System.out.println("Average relative error = " + avgRelativeError);	
    }
    
    public static void executeOnDevice(MyKernel kernel, float[] a, float[] b, float[] output, boolean[] selected) {
    	System.out.println("Execution mode: " + kernel.getExecutionMode());
        long t1_g = System.currentTimeMillis();
        // send parameters and execute, copy the OpenCL-hosted array back to RAM
        kernel.setA(a).setB(b).setOutput(output).setSelected(selected);
        kernel.put(a).put(b).put(output).put(selected);
        kernel.execute(a.length).get(output);
		long t2_g = System.currentTimeMillis();
		System.out.println("Device " + kernel.getExecutionMode() + " time diff: " + (t2_g-t1_g) + " ms");
    }

    public static MyRunnable[] executeOnJavaThreads(float[] a, float[] b, boolean selected) {
		int numProcessors = Runtime.getRuntime().availableProcessors();
		int chunkSize = a.length/numProcessors;
		System.out.println("number of processors/cores: " + numProcessors + ", CPU chunkSize: " + chunkSize);
		long t1 = System.currentTimeMillis();
		ExecutorService taskExecutor = Executors.newFixedThreadPool(numProcessors);
		MyRunnable[] tasks = new MyRunnable[numProcessors];
		for(int i = 0;i < numProcessors;i++) {
			MyRunnable task = new MyRunnable(a, b, selected, i*chunkSize, (i+1)*chunkSize);
			tasks[i] = task;
			taskExecutor.execute(task);
		}
		taskExecutor.shutdown();
		try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch(Exception ex) { }
		long t2 = System.currentTimeMillis();
		System.out.println("cpu time diff: " + (t2-t1) + " ms");
		return tasks;
    }

    public static final class MyStruct {
    	public float a;
    	public float b;
    	public float output;
    	
		public MyStruct(float a, float b) {
			super();
			this.a = a;
			this.b = b;
		}

/*		public float getA() {
			return a;
		}

		public void setA(float a) {
			this.a = a;
		}

		public float getB() {
			return b;
		}

		public void setB(float b) {
			this.b = b;
		}

		public float getOutput() {
			return output;
		}

		public void setOutput(float output) {
			this.output = output;
		}
    	*/
    }
    
    private static class MyStructuredKernel extends Kernel {
    	private MyStruct[] s;
    	public MyStructuredKernel(MyStruct[] s) {
    		this.s = s;
    	}
    	
		@Override
    	public void run() {
    		int i = getGlobalId();
    		s[i].output = sin(exp(cos(sin(s[i].a) * sin(s[i].b) + 1)));
    	}
    }

}
