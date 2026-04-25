8.2 Training methods
The overall training procedure is composed of three stages: (1) pretraining, (2) short–lead-time fine-tuning,
and (3) roll-out (long–lead-time) fine-tuning. We provide an overview for each of these stages in the following
paragraphs.
Training objective. Throughout pretraining and fine-tuning, we use the mean absolute error (MAE) as
our training objective L( ˆXt, Xt). Decomposing the predicted state ˆXt and ground-truth state Xt into
surface-level variables and atmospheric variables, ˆXt = ( ˆSt, ˆAt) and Xt = (St, At) (see Supplementary A),
the loss can be written as


$$
\mathcal{L}(\hat{X}^{t}, X^{t}) =
\frac{\gamma}{V_S + V_A}
\left[
\alpha
\left(
\sum_{k=1}^{V_S}
\frac{w_k^S}{H \times W}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\left|
\hat{S}_{k,i,j}^{t} - S_{k,i,j}^{t}
\right|
\right)
+
\beta
\left(
\sum_{k=1}^{V_A}
\frac{1}{C \times H \times W}
\sum_{c=1}^{C}
w_{k,c}^{A}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\left|
\hat{A}_{k,c,i,j}^{t} - A_{k,c,i,j}^{t}
\right|
\right)
\right],
$$


where wS
k is the weight associated with surface-level variable k, wA
k,c is the weight associated with atmospheric
variable k at pressure level c, α is a weight for the surface-level component of the loss, β is a weight for the
atmospheric component of the loss, and γ is a dataset-specific weight. See Supplementary D.1 for additional
details.
Pretraining methods. All models are pretrained for 150 k steps on 32 A100 GPUs, with a batch size of
one per GPU. We use a (half) cosine decay with a linear warm-up from zero for 1 k steps. The base learning
rate is 5e−4, which the schedule reduces by a factor 10 at the end of training. The optimizer we use is
AdamW [Loshchilov and Hutter, 2019]. We set the weight decay of AdamW to 5e−6. The only other form of
regularisation we use is drop path (i.e., stochastic depth) [Larsson et al., 2017], with the drop probability
set to 0.2. To make the model fit in memory, we use activation checkpointing for the backbone layers and
we shard all the model gradients across the GPUs. The model is trained using bf16 mixed precision. See
Supplementary D.2 for additional details.
Short–lead-time fine-tuning. After pretraining Aurora, for each task that we wish to adapt Aurora to,
we start by fine-tuning the entire architecture through one or two roll-out steps (depending on the task and
its memory constraints), see Supplementary D.3 for additional details.
Roll-out fine-tuning. To train very large Aurora models on long-term dynamics efficiently, even at high
resolutions, we develop a novel roll-out fine-tuning approach. Our approach uses Low Rank Adaptation
(LoRA) [Hu et al., 2021] to fine-tune all linear layers in the backbone’s self-attention operations, allowing
adaptation of very large models in a data and parameter efficient manner. To save memory, we employ the
“pushforward trick” [Brandstetter et al., 2022], which propagates gradients only through the last roll-out
step. Finally, to enable training at very large numbers of roll-out steps without compromising memory or
training speed, we use an in-memory replay buffer, inspired by deep reinforcement learning [Lin, 1992, Mnih
et al., 2015] (see Figure D2). The replay buffer samples initial conditions, computes predictions for the
next time step, adds predictions back to the replay buffer, and periodically refreshes the buffer with new
initial conditions from the dataset. For detailed roll-out protocols for each fine-tuning task, please refer to
Supplementary D.4



B.8 Extensions for wave forecasting
In Section 3, we fine-tune Aurora to forecast ocean surface waves. All adaptations of Aurora for wave
forecasting are described in this section.
Initialisation for new wave variables. All new wave variables are modelled as surface-level variables.
The new patch embeddings are all initialised to zero, except for the patch embeddings of 10UN and 10VN,
which are initialised to the patch embeddings of 10U and 10V learned during pretraining. Angle-valued variables. All angle-valued variables, which are MWD, MDWW, MDTS, MWD1, and
MWD2, are transformed with x 7 → ( sin (x), cos(x)) prior to running the encoder. The model has separate patch
embeddings for the sine and cosine components of angle-valued variables. After running the decoder, sine
and cosine components are transformed back to angles using (sin, cos) 7 → atan2(sin, cos). The angle-valued
variables have normalisation mean zero and normalisation scale one.
Density channels for missing data. Wave variables are undefined above land. In addition, wave variables
can be missing above water, for example in the case that there is sea ice or a swell component is just absent.
The model therefore must be able to handle data missing in the inputs and predict missing values in the
outputs. We accomplish this by, for every wave variable, incorporating a so-called density channel [Gordon
et al., 2020].
For a variable, the density channel is one if data are present and zero if data are absent. In the variable,
missing values are then simply set to zero after normalisation. We need to include such a density channel, as
otherwise the model would not be able to distinguish between a zero in the variable and a missing value.
The model treats the variable and the associated density channel as separate surface-level variables with
separate patch embeddings. The model also predicts the density channels, which is how the model can
predict that a variable is absent at a specific location. To bound the density channels to the range [0, 1], we
use the sigmoid function.
When running the model autoregressively, it will take its own predictions for density channels as inputs. In
this case, to avoid any distribution mismatch, we set all density channels to one wherever it is greater than
1
2 , and we set all density channels and associated data channels to zero wherever the density channels are
less than 1
2 .
Additional static variables. We include two additional static variables: the bathemetry, and a mask which
is one wherever HRES-WAM models wave variables (between latitudes -78° and 90°) and zero elsewhere.
The additional static variables are normalised to the range [0, 1].
Additional layer normalisation. When fine-tuning Aurora to the wave data, to stabilise training, we
apply a layer normalisation to the keys of the first level aggregation attention block and another layer
normalisation to the queries of the first level aggregation attention block. These layer normalisations are
applied before the keys and queries are split across multiple heads.

.9 Model hyperparameters and configurations
The 1.3B parameter Aurora model which we use in the main experiments instantiates the architecture as
follows. The embedding dimension in the encoder and first stage of the backbone is 512. This dimension
doubles at every subsequent stage of the backbone. The number of attention heads in the backbone is selected
such that the embedding dimension per head is 64 throughout the backbone. Due to the concatenation at
the end of the backbone, the embedding dimension in the decoder is 1024. In the Perceiver layers of the
encoder and decoder, we use an increased number of cross-attention heads (16), in order to give the model
fine-grained control over how the latent state of the atmosphere is constructed.
For the model scaling experiment in section Supplementary G, we instantiate smaller versions of this model
by reducing the number of backbone layers, the embedding dimension and the number of (cross) attention
heads, while always preserving the attention head dimension of 64 (Table B1).


D Training methods
The overall training procedure is composed of three stages: (1) pretraining, (2) short-lead-time fine-tuning,
(3) roll-out fine-tuning. We describe each of these stages in detail in the following subsections. 

D.1 Training objective
Throughout pretraining and fine-tuning, we use the mean absolute error (MAE) as our training objective
L( ˆXt, Xt). Decomposing the predicted state ˆXt and ground truth state Xt into surface-level variables and
atmospheric variables, ˆXt = ( ˆSt, ˆAt) and Xt = (St, At) (Supplementary A), the loss can be written as 

### D.1 Training objective

Throughout pretraining and fine-tuning, we use the mean absolute error (MAE) as our training objective
$\mathcal{L}(\hat{X}^{t}, X^{t})$. Decomposing the predicted state $\hat{X}^{t}$ and ground truth state $X^{t}$ into surface-level variables and atmospheric variables, $\hat{X}^{t} = (\hat{S}^{t}, \hat{A}^{t})$ and $X^{t} = (S^{t}, A^{t})$ (Supplementary A), the loss can be written as

$$
\mathcal{L}(\hat{X}^{t}, X^{t}) =
\frac{\gamma}{V_S + V_A}
\left[
\alpha
\left(
\sum_{k=1}^{V_S}
\frac{w_k^S}{H \times W}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\left|
\hat{S}_{k,i,j}^{t} - S_{k,i,j}^{t}
\right|
\right)
+
\beta
\left(
\sum_{k=1}^{V_A}
\frac{1}{C \times H \times W}
\sum_{c=1}^{C}
w_{k,c}^{A}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\left|
\hat{A}_{k,c,i,j}^{t} - A_{k,c,i,j}^{t}
\right|
\right)
\right],
$$


where wS
k denotes the weight associated with surface-level variable k and wA
k,c denotes the weight associated
with atmospheric variable k at pressure level c. The overall surface loss is weighted by α = 1
4 , while the
overall atmospheric loss is weighted by β = 1. Finally, the entire loss for a particular example is weighted by
a dataset weight γ, which allows us to upweight the datasets with higher fidelity such as ERA5 and GFS-T0.
Specifically, we use γERA5 = 2.0, γGFS-T0 = 1.5, and set the rest of the dataset weights to 1. When training,
we minimise the expected value of this loss computed over a mini-batch of samples.
Variable weighting for pretraining. During pretraining, we set wS
MSL = 1.5, wS
U10 = 0.77, wS
V10 = 0.66,
and wS
2T = 3.0. For the atmospheric variables, for all pressure levels c, we set wA
Z,c = 2.8, wA
Q,c = 0.78,
wA
T,c = 1.7, wA
U,c = 0.87, and wA
V,c = 0.6. The weights are chosen to balance the losses of the individual
variables and have been inspired by the weights used by Bi et al. [2023].
Variable weighting for concentrations of air pollutants. Air pollutants are extremely sparse. To
balance these variables with the meteorological variables, we require a radically different approach. For the
air pollution variables, we specify the weights such that the per-variable normalised MAE is roughly one.
For all air pollution variables, we compute the MAE for the persistence prediction of 12 h or 24 h, depending on
whether the model predicts the difference with respect to the state 12 h ago or 24 h ago (Supplementary B.7).
These persistence errors are computed on CAMS reanalysis data. We then set

$$
w_v^S = \frac{\mathrm{scale}_v}{\mathrm{persistence\ MAE}_v},
\qquad
w_{v,c}^A = \frac{\mathrm{scale}_{v,c}}{\mathrm{persistence\ MAE}_{v,c}}.
\tag{D13}
$$


ntuitively, multiplication by the scale first undoes the data normalisation and then dividing by the persistence
MAE brings the normalised MAE to have a value of approximately one.
Variable weighting for wave variables. When fine-tuning Aurora for wave forecasting, we tune the
weights to emphasise the more important and the more difficult variables. Specifically, we set sS
SWH = 2.0,
sS
MWP = 2.0, sS
MWD = 2.0, sS
PP1D = 4.0, sS
MPTS = 2.0, and sS
MDTS = 2.0.
Variable weighting for IFS T0 and analysis fine-tuning. During fine-tuning on IFS T0 0.25◦ and IFS
analysis 0.1◦, we slightly adjust the pretraining weights. For the surface-level variables, we set wS
MSL = 1.6,
wS
U10 = 0.77, wS
V10 = 0.66, and wS
2T = 3.5. For the atmospheric variables, for all pressure levels c, we set
wA
Z,c = 3.5, wA
Q,c = 0.8, wA
T,c = 1.7, wA
U,c = 0.87,and wA
V,c = 0.6.
Adjustments of the training loss for wave variables. For angle-valued variables, the MAE training
loss is computed over the sine and cosine components of the angle (see Supplementary B.8). The sine and
cosine components inherit the weighting of the original variables.
To deal with missing data, Aurora predicts the original variable along with a density channel (see Supplemen-
tary B.8). We compute the MAE over both the original variable and the density channel by also computing
the density channel for the target variable following exactly the procedure outlined Supplementary B.8. The
density channels inherit the weighting of the original variables. Note that the procedure sets all NaNs to
zero, meaning that the MAE can be computed without risking the loss to become NaN.

D.2 Pretraining methods
All models are pretrained for 150 k steps on 32 A100 GPUs, with a batch size of one per GPU. Our model
sees about 4.8 million frames after 150k steps of pretraining, which takes approximately two and a half
weeks. This roughly corresponds to 3 epochs over the C4 pretraining configuration that is outlined in
Supplementary G. We use a (half) cosine decay with a linear warm-up from zero for 1 k steps. The base
learning rate is 5e−4, which the schedule reduces by a factor 10× at the end of training. The optimizer we use
is AdamW [Loshchilov and Hutter, 2019]. We set the weight decay of AdamW to 5e−6. The only other form
of regularisation we use is drop path (i.e., stochastic depth) [Larsson et al., 2017], with the drop probability
set to 0.2. In order to make the model fit in memory, we use activation checkpointing for the backbone
layers and we shard all the model gradients across GPUs. The model is trained using bf16 mixed precision.
Aurora pretraining in comparison. The training of Aurora is comparable to GraphCast (four weeks on
32 Cloud TPU v4 devices, [Lam et al., 2023]) and faster than Pangu-Weather (about seven weeks using 64
GPUs, [Bi et al., 2023]), and roughly comparable to AIFS (about one week with 64 GPUs in total [Lang
et al., 2024]). This demonstrates that Aurora achieves its improved performance and additional capabilities
while maintaining training costs in line with other state-of-the-art models.
12-hour air pollution model. The 12-hour model used in the air pollution experiments was trained in
exactly the same manner as described above, but for 80.5 k steps instead of 150 k step

D.3 Short lead-time fine-tuning
For each task we wish to adapt the pretrained Aurora model to, we start by fine-tuning the entire architecture
through one or two roll-out steps (depending on the task and its memory constraints). In all cases, we use a
task-dependent hyperparameter selection, which we describe below.
HRES 0.25° T0. We fine-tune the weights of the entire model for 8 k training steps across 8 GPUs, with a
batch size of 1 per GPU. At each iteration, we perform two roll-out steps and backpropagate through both
of these steps. The model is optimised to minimise the MAE loss averaged across both roll-out steps. For
this regime, we use a 1 k step learning rate warm-up, followed by a constant learning rate of 5e−5. We use
the same weight decay as in pretraining and disable drop path. To ensure the model fits in memory for two
roll-out steps, we also use activation checkpointing for the encoder and the decoder, along with gradient
sharding as in pretraining.
HRES 0.1° analysis. We fine-tune the weights of the entire model for 12.5k steps across 8 GPUs, with a
batch size of 1 per GPU. Due to the increased memory constraints at this higher resolution, we train the
model only through a single step prediction. We use a 1k step learning rate warm-up, followed by a constant
learning rate of 2e−4. We set the weight decay to zero and disable drop path. To accommodate the higher
memory requirements, we use activation checkpointing for all the layers of the model and use sharding for
both the weights and the gradients.
CAMS 0.4° analysis. We train with single-step prediction, 12 h in this case, and the batch size is fixed to
1 per GPU. We use a linear warmup of 100 steps from zero, but we do not use a learning rate schedule after
that. We also use no weight decay, disable drop path, use activation checkpointing for all the layers of the
model, and use sharding for the weights and gradients.
Fine-tuning on CAMS analysis data proceeds in two steps. In the first step, we fine-tune on CAMS reanalysis
data using 16 GPUs for 22 k steps at the high learning rate and then for 14.5 k steps at the low learning rate.
The high learning rate is 1e−3 for the encoder patch embeddings of only the new pollution variables and
1e−4 for the rest of the the network. The low learning rate is 1e−4 for the encoder patch embeddings of only
the new pollution variables and 1e−4 for the rest of the network. To ensure maximum transfer from the
CAMS reanalysis data to the CAMS analysis data, the CAMS reanalysis data is regridded to the resolution
of CAMS analysis data, 0.4◦. In the second step, we fine-tune on CAMS analysis data using 8 GPUs for
7.5 k steps at the high learning rate and finally for 5.5 k steps at the low learning rate. The final model is
fine-tuned for 49.5 k steps in total.

HRES-WAM 0.25° analysis. We fine-tune the weights of the entire model for 14 k steps across 8 GPUs,
with a batch size of 1 per GPU. For the learning rate, we use a linear warmup of 500 steps to 1e−3 for
the encoder and decoder parameters of the new wave variables and 3e−4 for the rest of the network. The learning rate is then annealed to 1e−5 according to a cosine schedule for 30 k steps (the run was stopped at
14 k steps). Afterwards, the training data was restricted to the range Jul 2018–2021: on 5 June 2018, IFS
cycle 45r1 was implemented, which significantly changed the distribution of the data by coupling of the
three-dimensional ocean and atmosphere. For this adjusted training data range, we further fine-tune the
weights of the entire model for 10 k steps across 8 GPUs, with a batch size of 1 per GPU. For the learning
rate, we again use a linear warmup of 500 steps to 5e−5 for all parameters. The learning rate is then annealed
to 1e−5 according to a cosine schedule for 10 k steps. The final model is fine-tuned for 24 k steps in total

D.4 Roll-out fine-tuning
To ensure long-term multi-step dynamics, AI models typically fine-tune the model specifically for roll-outs.
Backpropagating through the autoregressive roll-outs for a large number of steps is unfeasible for a 1.3B
parameter model such as Aurora. This is particularly true at 0.1° resolution, where even a single step roll-out
requires close to the memory limit of an A100 GPU with 80GB of memory.
We use Low Rank Adaption (LoRA) [Hu et al., 2021] layers for roll-out fine-tuning all the linear layers
involved in the self-attention operations of the backbone. This allows us to take advantage of the large
model size and the fact that it can be easily adapted once pretrained. That is, for each linear transformation
W involved in the Swin self-attention layers, we learn low-rank matrices A, B to modulate the outputs of W
for an input x via W x + BAx. For more details, see Hu et al. [2021].
Furthermore, to avoid any memory increases compared to single-step fine-tuning, we use the “pushforward
trick” introduced in Brandstetter et al. [2022], where gradients are propagated only through the last roll-out
step. We run this at scale by using an in-memory replay buffer to avoid delays with generating long roll-outs
on each training step, similarly to how it is used in deep reinforcement learning [Lin, 1992, Mnih et al.,
2015] (Figure D2). At each training step, the model samples an initial condition from the replay buffer,
computes a prediction for the next time step, then adds this new prediction back to the replay buffer. We
periodically fetch fresh initial conditions from the dataset and add them to the replay buffer (i.e., the dataset
sampling period). This procedure allows the model to train at all roll-out steps without extra memory or
speed penalties.
HRES 0.25° analysis. We use 20 GPUs to fine-tune the LoRA layers for 13 k steps, each with a buffer
size of 200. This results in a total replay buffer size of 4000 samples. We use a dataset sampling period of 10
steps. To ensure the model learns to predict the early steps well (i.e., shorter lead-times) before attempting
to predict the later time steps, we use a schedule where for the first 5 k steps, we only keep predictions up to
4 days ahead in the buffer. The 4–10 day lead times are allowed in the buffer only after 5 k steps. We use a
constant learning rate of 5e−5.
HRES 0.1° analysis. Since the 0.1° data is 6.25× larger than 0.25° data, we use 32 GPUs with a buffer
size of 20 on each GPU. This is the maximum we can fit in the CPU memory of each node (i.e., 880 GB).
We use a dataset sampling period of 10 steps. We train the LoRA weights of the model for 6.25 k steps using
a constant learning rate of 5e−5.

CAMS 0.4° analysis. We use 16 GPUs with a buffer size of 200 on each GPU and a dataset sampling period
of 10 steps. We train the LoRA weights of the model for 6.5 k steps using a constant learning rate of 5e−5.
HRES-WAM 0.25° analysis. We use 8 GPUs with a buffer size of 100 on each GPU and a dataset
sampling period of 10 steps. We train the LoRA weights of the model for 6 k steps using a constant learning
rate of 5e−5.





