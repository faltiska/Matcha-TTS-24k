Most of the time when I tried to compile the model for training I ran into this exception:  
```
Error executing job with overrides: [] Traceback (most recent call last):   
File "matcha\train.py", line 151, in main     metric_dict, _ = train(cfg)   
File "matcha\utils\utils.py", line 117, in wrap     raise ex   
File "matcha\utils\utils.py", line 107, in wrap     metric_dict, object_dict = task_func(cfg=cfg)   
File "matcha\train.py", line 115, in train     trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=False)   
File "lightning\pytorch\trainer\trainer.py", line 584, in fit     call._call_and_handle_interrupt(   
File "lightning\pytorch\trainer\call.py", line 49, in _call_and_handle_interrupt     return trainer_fn(*args, **kwargs)   
File "lightning\pytorch\trainer\trainer.py", line 630, in _fit_impl     self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)   
File "lightning\pytorch\trainer\trainer.py", line 1079, in _run     results = self._run_stage()   
File "lightning\pytorch\trainer\trainer.py", line 1123, in _run_stage     self.fit_loop.run()   
File "lightning\pytorch\loops\fit_loop.py", line 217, in run     self.advance()   
File "lightning\pytorch\loops\fit_loop.py", line 465, in advance     self.epoch_loop.run(self._data_fetcher)   
File "lightning\pytorch\loops\training_epoch_loop.py", line 153, in run     self.advance(data_fetcher)   
File "lightning\pytorch\loops\training_epoch_loop.py", line 352, in advance     batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)   
File "lightning\pytorch\loops\optimization\automatic.py", line 192, in run     self._optimizer_step(batch_idx, closure)   
File "lightning\pytorch\loops\optimization\automatic.py", line 270, in _optimizer_step     call._call_lightning_module_hook(   
File "lightning\pytorch\trainer\call.py", line 177, in _call_lightning_module_hook     output = fn(*args, **kwargs)   
File "lightning\pytorch\core\module.py", line 1368, in optimizer_step     optimizer.step(closure=optimizer_closure)   
File "lightning\pytorch\core\optimizer.py", line 154, in step     step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)   
File "lightning\pytorch\strategies\strategy.py", line 239, in optimizer_step     return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)   
File "lightning\pytorch\plugins\precision\amp.py", line 76, in optimizer_step     return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)   
File "lightning\pytorch\plugins\precision\precision.py", line 123, in optimizer_step     return optimizer.step(closure=closure, **kwargs)   
File "torch\optim\optimizer.py", line 517, in wrapper     out = func(*args, **kwargs)   
File "torch\optim\optimizer.py", line 82, in _use_grad     ret = func(*args, **kwargs)   
File "torch\optim\adam.py", line 226, in step     loss = closure()   
File "lightning\pytorch\plugins\precision\precision.py", line 109, in _wrap_closure     closure_result = closure()   
File "lightning\pytorch\loops\optimization\automatic.py", line 146, in __call__     self._result = self.closure(*args, **kwargs)   
File "torch\utils\_contextlib.py", line 120, in decorate_context     return func(*args, **kwargs)   
File "lightning\pytorch\loops\optimization\automatic.py", line 131, in closure     step_output = self._step_fn()   
File "lightning\pytorch\loops\optimization\automatic.py", line 319, in _training_step     training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())   
File "lightning\pytorch\trainer\call.py", line 329, in _call_strategy_hook     output = fn(*args, **kwargs)   
File "lightning\pytorch\strategies\strategy.py", line 391, in training_step     return self.lightning_module.training_step(*args, **kwargs)   
File "torch\_dynamo\eval_frame.py", line 845, in compile_wrapper     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1   
File "torch\_inductor\compile_fx.py", line 990, in _compile_fx_inner     raise InductorError(e, currentframe()).with_traceback(   
File "torch\_inductor\compile_fx.py", line 974, in _compile_fx_inner     mb_compiled_graph = fx_codegen_and_compile(   
File "torch\_inductor\compile_fx.py", line 1695, in fx_codegen_and_compile     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)   
File "torch\_inductor\compile_fx.py", line 1505, in codegen_and_compile     compiled_module = graph.compile_to_module()   
File "torch\_inductor\graph.py", line 2319, in compile_to_module     return self._compile_to_module()   
File "torch\_inductor\graph.py", line 2325, in _compile_to_module     self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()   
File "torch\_inductor\graph.py", line 2264, in codegen     self.scheduler.codegen()   
File "torch\_inductor\scheduler.py", line 5197, in codegen     self._codegen_partitions()   
File "torch\_inductor\scheduler.py", line 5337, in _codegen_partitions     self._codegen(partition)   
File "torch\_inductor\scheduler.py", line 5435, in _codegen     self.get_backend(device).codegen_node(node)   
File "torch\_inductor\codegen\cuda_combined_scheduling.py", line 127, in codegen_node     return self._triton_scheduling.codegen_node(node)   
File "torch\_inductor\codegen\simd.py", line 1394, in codegen_node     coalesce_analysis = analyze_memory_coalescing(node)   
File "torch\_inductor\tiling_utils.py", line 650, in analyze_memory_coalescing     norm_read_writes = extract_normalized_read_writes(fused_node)   
File "torch\_inductor\tiling_utils.py", line 482, in extract_normalized_read_writes     if any(   
File "torch\_inductor\tiling_utils.py", line 483, in <genexpr>     (isinstance(var, sympy.Expr) and not var.is_constant())   
File "sympy\core\expr.py", line 724, in is_constant     b = expr._random(None, -1, 0, 1, 0)   
File "sympy\core\expr.py", line 562, in _random     nmag = abs(self.evalf(2, subs=reps))   
File "sympy\core\evalf.py", line 1654, in evalf     result = evalf(self, prec + 4, options)   
File "sympy\core\evalf.py", line 1489, in evalf     r = rf(x, prec, options)   
File "sympy\core\evalf.py", line 602, in evalf_add     terms = [evalf(arg, prec + 10, options) for arg in v.args]   
File "sympy\core\evalf.py", line 602, in <listcomp>     terms = [evalf(arg, prec + 10, options) for arg in v.args]   
File "sympy\core\evalf.py", line 1489, in evalf     r = rf(x, prec, options)   
File "sympy\core\evalf.py", line 650, in evalf_mul     result = evalf(arg, prec, options)   
File "sympy\core\evalf.py", line 1493, in evalf     x = x.subs(evalf_subs(prec, options['subs']))   
File "sympy\core\basic.py", line 1171, in subs     rv = rv._subs(old, new, **kwargs)   
File "sympy\core\cache.py", line 72, in wrapper     retval = cfunc(*args, **kwargs)   
File "sympy\core\basic.py", line 1285, in _subs     rv = fallback(self, old, new)   
File "sympy\core\basic.py", line 1262, in fallback     rv = self.func(*args)   
File "sympy\core\cache.py", line 72, in wrapper     retval = cfunc(*args, **kwargs)   
File "sympy\core\function.py", line 450, in __new__     return cls._new_(*args, **options)  # type: ignore   
File "sympy\core\function.py", line 472, in _new_     result = super().__new__(cls, *args, **options)   
File "sympy\core\cache.py", line 72, in wrapper     retval = cfunc(*args, **kwargs)   
File "sympy\core\function.py", line 309, in __new__     evaluated = cls.eval(*args)   
File "torch\utils\_sympy\functions.py", line 495, in eval     assert p >= 0, p torch._inductor.exc.InductorError: AssertionError: -43411219694767/200000000000000
```
which I think is a bug in torch inductor.

I can "fix it" by modifying .venv/Lib/site-packages/sympy/core/expr.py, putting:
```
b = expr._random(None, 0, 0, 1, 0) 
```
instead of:
```
b = expr._random(None, -1, 0, 1, 0) 
```
at line 724 and 728.

I've logged a bug for it: https://github.com/pytorch/pytorch/issues/170550
It is already fixed in the main branch and included in torch cuda nightly
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

