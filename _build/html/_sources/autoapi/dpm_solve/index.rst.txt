dpm_solve
=========

.. py:module:: dpm_solve






Module Contents
---------------

.. py:class:: NoiseScheduleVP(schedule='discrete', betas=None, alphas_cumprod=None, continuous_beta_0=0.1, continuous_beta_1=20.0, dtype=torch.float32)

   
   Create a wrapper class for the forward SDE (VP type).

   ***
   Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
           We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
   ***

   The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
   We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
   Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

       log_alpha_t = self.marginal_log_mean_coeff(t)
       sigma_t = self.marginal_std(t)
       lambda_t = self.marginal_lambda(t)

   Moreover, as lambda(t) is an invertible function, we also support its inverse function:

       t = self.inverse_lambda(lambda_t)

   ===============================================================

   We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

   1. For discrete-time DPMs:

       For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
           t_i = (i + 1) / N
       e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
       We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

       Args:
           betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
           alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

       Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

       **Important**:  Please pay special attention for the args for `alphas_cumprod`:
           The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
               q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
           Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
               alpha_{t_n} = \sqrt{\hat{alpha_n}},
           and
               log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


   2. For continuous-time DPMs:

       We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
       schedule are the default settings in DDPM and improved-DDPM:

       Args:
           beta_min: A `float` number. The smallest beta for the linear schedule.
           beta_max: A `float` number. The largest beta for the linear schedule.
           cosine_s: A `float` number. The hyperparameter in the cosine schedule.
           cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
           T: A `float` number. The ending time of the forward process.

   ===============================================================

   :param schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.

   :returns: A wrapper object of the forward SDE (VP type).

   ===============================================================

   Example:

   # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
   >>> ns = NoiseScheduleVP('discrete', betas=betas)

   # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
   >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

   # For continuous-time DPMs (VPSDE), linear schedule:
   >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)



   .. py:attribute:: schedule
      :value: 'discrete'



   .. py:method:: marginal_log_mean_coeff(t)

      Compute log(alpha_t) of a given continuous-time label t in [0, T].



   .. py:method:: marginal_alpha(t)

      Compute alpha_t of a given continuous-time label t in [0, T].



   .. py:method:: marginal_std(t)

      Compute sigma_t of a given continuous-time label t in [0, T].



   .. py:method:: marginal_lambda(t)

      Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].



   .. py:method:: inverse_lambda(lamb)

      Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.



.. py:function:: model_wrapper(model, noise_schedule, model_type='noise', model_kwargs={}, guidance_type='uncond', condition=None, unconditional_condition=None, guidance_scale=1.0, classifier_fn=None, classifier_kwargs={})

   Create a wrapper function for the noise prediction model.

   DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
   firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

   We support four types of the diffusion model by setting `model_type`:

       1. "noise": noise prediction model. (Trained by predicting noise).

       2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

       3. "v": velocity prediction model. (Trained by predicting the velocity).
           The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

           [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
               arXiv preprint arXiv:2202.00512 (2022).
           [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
               arXiv preprint arXiv:2210.02303 (2022).

       4. "score": marginal score function. (Trained by denoising score matching).
           Note that the score function and the noise prediction model follows a simple relationship:
           ```
               noise(x_t, t) = -sigma_t * score(x_t, t)
           ```

   We support three types of guided sampling by DPMs by setting `guidance_type`:
       1. "uncond": unconditional sampling by DPMs.
           The input `model` has the following format:
           ``
               model(x, t_input, **model_kwargs) -> noise | x_start | v | score
           ``

       2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
           The input `model` has the following format:
           ``
               model(x, t_input, **model_kwargs) -> noise | x_start | v | score
           ``

           The input `classifier_fn` has the following format:
           ``
               classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
           ``

           [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
               in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

       3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
           The input `model` has the following format:
           ``
               model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
           ``
           And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

           [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
               arXiv preprint arXiv:2207.12598 (2022).


   The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
   or continuous-time labels (i.e. epsilon to T).

   We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
   ``
       def model_fn(x, t_continuous) -> noise:
           t_input = get_model_input_time(t_continuous)
           return noise_pred(model, x, t_input, **model_kwargs)
   ``
   where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

   ===============================================================

   :param model: A diffusion model with the corresponding format described above.
   :param noise_schedule: A noise schedule object, such as NoiseScheduleVP.
   :param model_type: A `str`. The parameterization type of the diffusion model.
                      "noise" or "x_start" or "v" or "score".
   :param model_kwargs: A `dict`. A dict for the other inputs of the model function.
   :param guidance_type: A `str`. The type of the guidance for sampling.
                         "uncond" or "classifier" or "classifier-free".
   :param condition: A pytorch tensor. The condition for the guided sampling.
                     Only used for "classifier" or "classifier-free" guidance type.
   :param unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                                   Only used for "classifier-free" guidance type.
   :param guidance_scale: A `float`. The scale for the guided sampling.
   :param classifier_fn: A classifier function. Only used for the classifier guidance.
   :param classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.

   :returns: A noise prediction model that accepts the noised data and the continuous time as the inputs.


.. py:class:: DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++', correcting_x0_fn=None, correcting_xt_fn=None, thresholding_max_val=1.0, dynamic_thresholding_ratio=0.995)

   
   Construct a DPM-Solver.

   We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

   We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
   can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
   dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
   DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
   DPMs (such as stable-diffusion).

   To support advanced algorithms in image-to-image applications, we also support corrector functions for
   both x0 and xt.

   :param model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                    ``
                    def model_fn(x, t_continuous):
                        return noise
                    ``
                    The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
   :param noise_schedule: A noise schedule object, such as NoiseScheduleVP.
   :param algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
   :param correcting_x0_fn: A `str` or a function with the following format:
                            ```
                            def correcting_x0_fn(x0, t):
                                x0_new = ...
                                return x0_new
                            ```
                            This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                            ```
                            x0_pred = data_pred_model(xt, t)
                            if correcting_x0_fn is not None:
                                x0_pred = correcting_x0_fn(x0_pred, t)
                            xt_1 = update(x0_pred, xt, t)
                            ```
                            If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
   :param correcting_xt_fn: A function with the following format:
                            ```
                            def correcting_xt_fn(xt, t, step):
                                x_new = ...
                                return x_new
                            ```
                            This function is to correct the intermediate samples xt at each sampling step. e.g.,
                            ```
                            xt = ...
                            xt = correcting_xt_fn(xt, t, step)
                            ```
   :param thresholding_max_val: A `float`. The max value for thresholding.
                                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
   :param dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                                      Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

   [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
       Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
       with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.


   .. py:attribute:: model


   .. py:attribute:: noise_schedule


   .. py:attribute:: algorithm_type
      :value: 'dpmsolver++'



   .. py:attribute:: correcting_xt_fn
      :value: None



   .. py:attribute:: dynamic_thresholding_ratio
      :value: 0.995



   .. py:attribute:: thresholding_max_val
      :value: 1.0



   .. py:method:: dynamic_thresholding_fn(x0, t)

      The dynamic thresholding method.



   .. py:method:: noise_prediction_fn(x, t)

      Return the noise prediction model.



   .. py:method:: data_prediction_fn(x, t)

      Return the data prediction model (with corrector).



   .. py:method:: model_fn(x, t)

      Convert the model to the noise prediction model or the data prediction model.



   .. py:method:: get_time_steps(skip_type, t_T, t_0, N, device)

      Compute the intermediate time steps for sampling.

      :param skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                        - 'logSNR': uniform logSNR for the time steps.
                        - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                        - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
      :param t_T: A `float`. The starting time of the sampling (default is T).
      :param t_0: A `float`. The ending time of the sampling (default is epsilon).
      :param N: A `int`. The total number of the spacing of the time steps.
      :param device: A torch device.

      :returns: A pytorch tensor of the time steps, with the shape (N + 1,).



   .. py:method:: get_orders_and_timesteps_for_singlestep_solver(steps, order, skip_type, t_T, t_0, device)

      Get the order of each step for sampling by the singlestep DPM-Solver.

      We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
      Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
          - If order == 1:
              We take `steps` of DPM-Solver-1 (i.e. DDIM).
          - If order == 2:
              - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
              - If steps % 2 == 0, we use K steps of DPM-Solver-2.
              - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
          - If order == 3:
              - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
              - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
              - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
              - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

      ============================================
      :param order: A `int`. The max order for the solver (2 or 3).
      :param steps: A `int`. The total number of function evaluations (NFE).
      :param skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                        - 'logSNR': uniform logSNR for the time steps.
                        - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                        - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
      :param t_T: A `float`. The starting time of the sampling (default is T).
      :param t_0: A `float`. The ending time of the sampling (default is epsilon).
      :param device: A torch device.

      :returns: *orders* -- A list of the solver order of each step.



   .. py:method:: denoise_to_zero_fn(x, s)

      Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.



   .. py:method:: dpm_solver_first_update(x, s, t, model_s=None, return_intermediate=False)

      DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param s: A pytorch tensor. The starting time, with the shape (1,).
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param model_s: A pytorch tensor. The model function evaluated at time `s`.
                      If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
      :param return_intermediate: A `bool`. If true, also return the model value at time `s`.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: singlestep_dpm_solver_second_update(x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver')

      Singlestep solver DPM-Solver-2 from time `s` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param s: A pytorch tensor. The starting time, with the shape (1,).
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param r1: A `float`. The hyperparameter of the second-order solver.
      :param model_s: A pytorch tensor. The model function evaluated at time `s`.
                      If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
      :param return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: singlestep_dpm_solver_third_update(x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver')

      Singlestep solver DPM-Solver-3 from time `s` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param s: A pytorch tensor. The starting time, with the shape (1,).
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param r1: A `float`. The hyperparameter of the third-order solver.
      :param r2: A `float`. The hyperparameter of the third-order solver.
      :param model_s: A pytorch tensor. The model function evaluated at time `s`.
                      If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
      :param model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                       If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
      :param return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type='dpmsolver')

      Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param model_prev_list: A list of pytorch tensor. The previous computed model values.
      :param t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type='dpmsolver')

      Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param model_prev_list: A list of pytorch tensor. The previous computed model values.
      :param t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: singlestep_dpm_solver_update(x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None)

      Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param s: A pytorch tensor. The starting time, with the shape (1,).
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
      :param return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
      :param r1: A `float`. The hyperparameter of the second-order or third-order solver.
      :param r2: A `float`. The hyperparameter of the third-order solver.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver')

      Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

      :param x: A pytorch tensor. The initial value at time `s`.
      :param model_prev_list: A list of pytorch tensor. The previous computed model values.
      :param t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
      :param t: A pytorch tensor. The ending time, with the shape (1,).
      :param order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.

      :returns: *x_t* -- A pytorch tensor. The approximated solution at time `t`.



   .. py:method:: dpm_solver_adaptive(x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-05, solver_type='dpmsolver')

      The adaptive step size solver based on singlestep DPM-Solver.

      :param x: A pytorch tensor. The initial value at time `t_T`.
      :param order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
      :param t_T: A `float`. The starting time of the sampling (default is T).
      :param t_0: A `float`. The ending time of the sampling (default is epsilon).
      :param h_init: A `float`. The initial step size (for logSNR).
      :param atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
      :param rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
      :param theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
      :param t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                    current time and `t_0` is less than `t_err`. The default setting is 1e-5.
      :param solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                          The type slightly impacts the performance. We recommend to use 'dpmsolver' type.

      :returns: *x_0* -- A pytorch tensor. The approximated solution at time `t_0`.

      [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.



   .. py:method:: add_noise(x, t, noise=None)

      Compute the noised input xt = alpha_t * x + sigma_t * noise.

      :param x: A `torch.Tensor` with shape `(batch_size, *shape)`.
      :param t: A `torch.Tensor` with shape `(t_size,)`.

      :returns: xt with shape `(t_size, batch_size, *shape)`.



   .. py:method:: inverse(x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver', atol=0.0078, rtol=0.05, return_intermediate=False)

      Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
      For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.



   .. py:method:: sample(x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver', atol=0.0078, rtol=0.05, return_intermediate=False)

      Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

      =====================================================

      We support the following algorithms for both noise prediction model and data prediction model:
          - 'singlestep':
              Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver.
              We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
              The total number of function evaluations (NFE) == `steps`.
              Given a fixed NFE == `steps`, the sampling procedure is:
                  - If `order` == 1:
                      - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                  - If `order` == 2:
                      - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                      - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                      - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                  - If `order` == 3:
                      - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                      - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                      - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                      - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
          - 'multistep':
              Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
              We initialize the first `order` values by lower order multistep solvers.
              Given a fixed NFE == `steps`, the sampling procedure is:
                  Denote K = steps.
                  - If `order` == 1:
                      - We use K steps of DPM-Solver-1 (i.e. DDIM).
                  - If `order` == 2:
                      - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                  - If `order` == 3:
                      - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
          - 'singlestep_fixed':
              Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
              We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
          - 'adaptive':
              Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
              We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
              You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
              (NFE) and the sample quality.
                  - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                  - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

      =====================================================

      Some advices for choosing the algorithm:
          - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
              Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
              e.g., DPM-Solver:
                  >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                  >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                          skip_type='time_uniform', method='singlestep')
              e.g., DPM-Solver++:
                  >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                  >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                          skip_type='time_uniform', method='singlestep')
          - For **guided sampling with large guidance scale** by DPMs:
              Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
              e.g.
                  >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                  >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                          skip_type='time_uniform', method='multistep')

      We support three types of `skip_type`:
          - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
          - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
          - 'time_quadratic': quadratic time for the time steps.

      =====================================================
      :param x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
      :param steps: A `int`. The total number of function evaluations (NFE).
      :param t_start: A `float`. The starting time of the sampling.
                      If `T` is None, we use self.noise_schedule.T (default is 1.0).
      :param t_end: A `float`. The ending time of the sampling.
                    If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                    e.g. if total_N == 1000, we have `t_end` == 1e-3.
                    For discrete-time DPMs:
                        - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                    For continuous-time DPMs:
                        - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
      :param order: A `int`. The order of DPM-Solver.
      :param skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
      :param method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
      :param denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                              Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                              This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                              score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                              for diffusion models sampling by diffusion SDEs for low-resolutional images
                              (such as CIFAR-10). However, we observed that such trick does not matter for
                              high-resolutional images. As it needs an additional NFE, we do not recommend
                              it for high-resolutional images.
      :param lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                                (especially for steps <= 10). So we recommend to set it to be `True`.
      :param solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
      :param atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
      :param rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
      :param return_intermediate: A `bool`. Whether to save the xt at each step.
                                  When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.

      :returns: *x_end* -- A pytorch tensor. The approximated solution at time `t_end`.



.. py:function:: interpolate_fn(x, xp, yp)

   A piecewise linear function y = f(x), using xp and yp as keypoints.
   We implement f(x) in a differentiable way (i.e. applicable for autograd).
   The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

   :param x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
   :param xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
   :param yp: PyTorch tensor with shape [C, K].

   :returns: The function values f(x), with shape [N, C].


.. py:function:: expand_dims(v, dims)

   Expand the tensor `v` to the dim `dims`.

   :param `v`: a PyTorch tensor with shape [N].
   :param `dim`: a `int`.

   :returns: a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.


