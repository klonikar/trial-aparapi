package trial.aparapi;

import com.amd.aparapi.Kernel;

//javac -g -cp lib/aparapi.jar -d bin src/trial/aparapi/Reducer.java
//windows:
//java -cp lib/aparapi.jar;bin -Djava.library.path=lib trial.aparapi.Reducer
//linux:
//java -cp lib/aparapi.jar:bin -Djava.library.path=lib trial.aparapi.Reducer

public class Reducer {

    public static class ReducerKernel extends Kernel {
    	private long[] a;
    	private int[] iterationNo;
    	private long[] output;
    	
    	public ReducerKernel() {
			super();
			setExplicit(true);
		}

    	public ReducerKernel setA(long[] val) {
    		this.a = val;
    		return this;
    	}

    	public ReducerKernel setIterationNo(int[] val) {
    		this.iterationNo = val;
    		return this;
    	}

    	public ReducerKernel setOutput(long[] val) {
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

    public static void executeOnDevice() {
        ReducerKernel kernel = new ReducerKernel();
        long[] vector = new long[1 << 23];
        for(int i = 0;i < vector.length;i++) {
        	vector[i] = (long) (i+1);
        }
    	//kernel.setExecutionMode(EXECUTION_MODE.JTP);
    	System.out.println("Execution mode: " + kernel.getExecutionMode());
    	long t1 = System.currentTimeMillis();
    	long sum = 0;
    	for(int i = 0;i < vector.length;i++) {
    		sum += vector[i];
    	}
    	long t2 = System.currentTimeMillis();
    	System.out.println("Raw computed sum: " + ((long) sum) + ", time " + (t2-t1) + "ms");
    	// compute a dummy sum to compile the kernel
    	long[] output = new long[] { 0 };
    	{
    		long[] a = new long[] { 1 };
    		int[] iterNo = new int[] { 0, a.length, 2, 1, 1 };
    		kernel.setA(a).setOutput(output).setIterationNo(iterNo).put(a).put(output).put(iterNo).execute(1);
    	}
        long t1_g = System.currentTimeMillis();
        // send parameters and execute, copy the OpenCL-hosted array back to RAM
        int localBatchSize = 8;
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
        	long t1_step_g = System.currentTimeMillis();
	        kernel.setIterationNo(iterationNo).put(iterationNo).execute(kernelRange);
	        long t2_step_g = System.currentTimeMillis();
	        System.out.println("Device " + kernel.getExecutionMode() + " time diff: " + (t2_step_g-t1_step_g) + " ms");
        }
        kernel.get(output);
		long t2_g = System.currentTimeMillis();
		System.out.println("Device " + kernel.getExecutionMode() + " time diff: " + (t2_g-t1_g) + " ms");
		System.out.println("Sum: " + ((long) output[0]));
    }

	public static void main(String[] args) {
        executeOnDevice();
	}

}
