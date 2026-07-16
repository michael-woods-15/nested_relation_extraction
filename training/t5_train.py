import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
 
 
def select_precision():
    """Pick the best available mixed-precision mode.
 
    Prefers bf16 (no loss scaling needed, generally more stable) on hardware
    that supports it, falls back to fp16 on other CUDA devices, and disables
    both on CPU.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}
 
 
def build_trainer(model, tokenizer, tokenized_datasets, compute_metrics, *, output_dir, learning_rate, train_batch_size,
    eval_batch_size, num_epochs, generation_max_length, logging_steps, metric_for_best_model, greater_is_better, 
    bf16=None, fp16=None):
    
    if bf16 is None and fp16 is None:
        precision = select_precision()
        bf16, fp16 = precision["bf16"], precision["fp16"]
    else:
        bf16 = bool(bf16)
        fp16 = bool(fp16)
 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
 
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )
 
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )