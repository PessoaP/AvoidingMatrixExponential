# AvoidingMatrixExponential

This is the code associated to our article ''Avoiding matrix exponentials for large transition rate matrices'' available as a preprint on [arXiv](https://arxiv.org/abs/2312.05647)

## Simulation
For usage in simulations one can type, for example, 
`python 1_1S_Simulations.py 50`
where the 50 above can be susbstituted by the value of $\beta$ according to Section 3.1.1 in the manuscript. 

The two states system (Section 3.1.2) can be simulated as
`python 4_2S_Simulations.py 100`

And for the autoregulatory gene network (Section 3.1.3) is simulated by typing
`python 7-STS_Simulations.py 10. .5`
where the 10. and .5 corresponds to $\beta_R$ and $\beta_P in the manuscript.

## Time Benchmark
If the simulated data is ready, the time benchmark for most methods can be done by
`python 2-0_1S_time_benchmark.py 100 rk`
where one must use the same $\beta$ (100 above) as in the simulation and the last piece (`rk` above) must be subsituted by the method being benchmarked (`rk` for Runge-Kutta, `kry` for Krylov, `rmjp` for R-MJP, and `me` for naive matrix exponential).
In order to benchmark J-MJP, which requires a significantly different code, we use
`python 2-4_1S_JMJP_time_benchmark.py 100`


The equivalent bash for the two states and autoregulatory are, respectively, 

`python 5-0_2S_time_benchmark.py 100 rk`

`python 8-0_STS_time_benchmark.py 100 rk`


and for JMJP we can, respectively, use

`python 5-4_2S_JMJP_time_benchmark.py 100 rk`

`python 8-4_STS_JMJP_time_benchmark.py 100 rk`

## Inference

Finally the inference (needed in Fig. 4, Fig.5, and SI-B) is done by, 

`python 3-0_1S_Gibbs.py 100 rk`

by replacing the $\beta$ and the method as for the time benchmark. 

The equivalent 

`python 6-0_2S_Gibbs.py 100 rk`

`python 9-0_STS_Gibbs.py 100 rk`

and for JMJP use

`python 6-0_2S_JMJP_Gibbs.py 100 rk`

`python 9-0_STS_JMJP_Gibbs.py 100 rk`
