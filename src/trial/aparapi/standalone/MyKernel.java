package trial.aparapi.standalone;

import com.amd.aparapi.Kernel;

public class MyKernel extends Kernel {
	private float[] quantity;
	private float[] discount;
	private float[] output;
	
	public MyKernel() {
		super();
	}

	public MyKernel setQ(float[] a) {
		this.quantity = a;
		return this;
	}

	public MyKernel setD(float[] b) {
		this.discount = b;
		return this;
	}

	public MyKernel setOutput(float[] output) {
		this.output = output;
		return this;
	}

	@Override
	public void run() {
		int i = getGlobalId();
		float DISCOUNT = discount[i];
		float QUANTITY = quantity[i];
		output[i] = exp(sin(cos(sin(1.0f+exp(cos(sin(sin(sqrt(8.0f+exp(1.0f+sin(pow(QUANTITY, 4.5f) - exp(DISCOUNT))))+2.0f)*exp(1.0f+QUANTITY)*cos(1.0f+exp(DISCOUNT))+2.0f))+5.0f)))))*sin(cos(cos(exp(1.0f+exp(sin(sin(sin(sqrt(8.0f+exp(1.0f+cos(pow(QUANTITY, sqrt(2.0f)) - sin(DISCOUNT))))+2.0f)*exp(2.0f+QUANTITY)*cos(3+exp(DISCOUNT*2.0f))+2.0f))+5.0f)))));
		//output[i] = pow(QUANTITY, (float) 4.5) - exp(DISCOUNT);
	}
}
