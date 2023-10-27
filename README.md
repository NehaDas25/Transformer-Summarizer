# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- GitHub: <ins> NehaDas25 </ins>

## Features

- Not Implemented:
  - what features have been implemented

<br><br>

- Implemented:
  - PART 2: Summarization with Transformer
  - PART 2.1 - Dot Product Attention
    - Exercise 1 - DotProductAttention
      - Implemented the DotProductAttention() that takes in query, key, value, mask.
      - Save depth/dimension of the query embedding for scaling down the dot product.
      - Calculated scaled query key dot product according to formula Attention (𝑄,𝐾,𝑉)=softmax(𝑄𝐾^𝑇/√𝑑𝑘+𝑀)𝑉.
      - Apply the mask.
      - Implementated the Softmax using trax.fastmath.logsumexp of masked_qkT to avoid underflow by division by large numbers.
      - Taken exponential of dots minus logsumexp to get softmax using jnp.exp().
      - Multiply dots by value to get self-attention using jnp.matmul().
      - This passed all the test cases.
  
  - PART 2.2 - Causal Attention
    - Exercise 2a - compute_attention_heads
      - Inside a compute_attention_heads_closure(n_heads, d_head) function, implemented the function compute_attention_heads() that takes x as input.
      - Size of the x's batch dimension should be taken.
      - Length of the sequence should be size of x's first dimension without counting the batch dim.
      - Reshaped x using jnp.reshape() n_batch, seqlen, n_heads*d_head -> n_batch, seqlen, n_heads, d_head.
      - Transpose x using jnp.transpose() n_batch, seqlen, n_heads, d_head -> n_batch, n_heads, seqlen, d_head.Noted that the values within the tuple are the indexes of the dimensions of x and must rearrange them.
      - Reshaped x using jnp.reshape() n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head.
      - This passed all the unit-test cases.

    - Exercise 2b - dot_product_self_attention
      - Inside a compute_attention_heads_closure(n_heads, d_head) function, implemented the function dot_product_self_attention() that takes q, k, v as input.
      - Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
      - Notice that 1's and 0's get casted to True/False by setting dtype to jnp.bool_.
      - Used jnp.tril() - lower triangle of an array and jnp.ones().
      - This passed all the unit-test cases.
    
    - Exercise 2c - compute_attention_output
      - Inside a compute_attention_heads_closure(n_heads, d_head) function, implemented the function compute_attention_output() that takes x as input.
      - Length of the sequence should be size of x's first dimension without counting the batch dim.
      - Reshape x using jnp.reshape() to shape (n_batch, n_heads, seqlen, d_head).
      - Transpose x using jnp.transpose() to shape (n_batch, seqlen, n_heads, d_head).
      - This passed all the unit-test cases.

    - Exercise 2d - CausalAttention
      - Implemented the CausalAttention() that takes in d_feature,n_heads,compute_attention_heads_closure(),dot_product_self_attention(), compute_attention_output_closure(). 
      - This model returns the causal attention through a  𝑡𝑙.𝑆𝑒𝑟𝑖𝑎𝑙() withe the following:
       1. tl.Branch() : consisting of 3 [tl.Dense(d_feature), ComputeAttentionHeads] to account for the queries, keys, and values.
       2. tl.Fn(): Takes in dot_product_self_attention function and uses it to compute the dot product using 𝑄,𝐾,𝑉 and also takes in compute_attention_output_closure to allow for parallel computing.
       3. tl.Dense(): Final Dense layer, with dimension d_feature.
      - This passed all the test cases as well.
  
  - PART 2.3 - Transformer Decoder Block
    - Exercise 3 - DecoderBlock
      - Implemented the function DecoderBlock() that takes in d_model, d_ff, n_heads,dropout, mode, ff_activation.
      - Created masked multi-head attention block using CausalAttention function.
      - Created feed-forward block (list) with two dense layers with dropout and input normalized.
       1. Normalize layer inputs using tl.LayerNorm().
       2. Add first feed forward (dense) layer using tl.Dense(d_ff).
       3. Add dropout with rate and mode specified using tl.Dropout(rate = dropout, mode=mode).
       4. Add second feed forward layer using tl.Dense(d_model).
      - Added list of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
      - This passed the unit-test cases as well.

  - PART 2.4 - Transformer Language Model
    - Exercise 4 - TransformerLM
      - Implemented the function TransformerLM() that takes in vocab_size=33300,d_model=512,d_ff=2048,n_layers=6,n_heads=8,dropout=0.1,max_len=4096,mode='train',ff_activation=tl.Relu.
      - Returns a Transformer language model.
      - Embedding inputs and positional encoder.
      - Created stack (list) of decoder blocks with n_layers with necessary parameters.
      - Create the complete model as written in the figure:tl.Serial: takes in the following layers or lists of layers:
       1. tl.ShiftRight: : shift the tensor to the right by padding on axis 1.
       2. positional_encoder : encodes the text positions.
       3. decoder_blocks : the ones you created.
       4. tl.LayerNorm : a layer norm.
       5. tl.Dense : takes in the vocab_size.
       6. tl.LogSoftmax : to predict 
      - This passed all the unit-test cases as well.
 
  - PART 3: Training
    - Exercise 5 - training_loop
      - Implement the train_model program below to train the neural network above.
      - Create the train task by calling trax.supervised.training.TrainTask and pass in the following:
       1. labeled_data = train_gen
       2. loss_layer = tl.CrossEntropyLoss()
       3. optimizer = trax.optimizers.Adam(0.01)
       4. lr_schedule = lr_schedule
      - Create the eval task by calling trax.supervised.training.EvalTask and pass in the following:
       1. labeled_data = eval_gen
       2. metrics = tl.CrossEntropyLoss() and tl.Accuracy()
      - Create the training loop by calling trax.supervised.Training.Loop and pass in the following:
       1. TransformerLM
       2. train_task
       3. eval_tasks = [eval_task]
       4. output_dir = output_dir
      - This passed all the test cases as well.

  - PART 5 - Testing with your Own Input
    - Exercise 6 - next_symbol
      -  Implemented the next symbol function that takes in the cur_output_tokens and the trained model to return the the index of the next word.
      - current output tokens length.
      - Calculated the minimum power of 2 big enough to store token_length and add 1 to token_length so np.log2() doesn't receive 0 when token_length is 0.
      - Fill cur_output_tokens with 0's until it reaches padded_length.
      - model expects a tuple containing two padded tensors (with batch),and to get log_probs we need to index output wih 0 in the first dim and token_length in the second dim and all of the entries for the last dim.
      - This passed all the test cases as well.

  - PART 5.1 - Greedy Decoding
    - Exercise 7 - greedy_decode 
      - Implement the function greedy_decode() that takes in input_sentence, model, next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize, vocab_dir='vocab_dir/', verbose=False.
      - Use tokenize().
      - Get next symbol that is  cur_output = next_symbol(cur_output_tokens, model).
      - Appended next symbol to original sentence.
      - Appended next symbol to generated sentence.
      - This passed all the test cases as well.


<br><br>

- Partly implemented:
  - what features have not been implemented

<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Assignment is very good. Gives a thorough understanding of the multi head attention, encoder-decoder,greedy_encoding().


## Output

### output:

<pre>
<br/><br/>
Out[5] - 

Single example mask:

 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

Out[6] - 

Single example:

 By. Associated Press. PUBLISHED:. 14:11 EST, 25 October 2013. |.
UPDATED:. 15:36 EST, 25 October 2013. The bishop of the Fargo Catholic
Diocese in North Dakota has exposed potentially hundreds of church
members in Fargo, Grand Forks and Jamestown to the hepatitis A virus
in late September and early October. The state Health Department has
issued an advisory of exposure for anyone who attended five churches
and took communion. Bishop John Folda (pictured) of the Fargo Catholic
Diocese in North Dakota has exposed potentially hundreds of church
members in Fargo, Grand Forks and Jamestown to the hepatitis A. State
Immunization Program Manager Molly Howell says the risk is low, but
officials feel it's important to alert people to the possible
exposure. The diocese announced on Monday that Bishop John Folda is
taking time off after being diagnosed with hepatitis A. The diocese
says he contracted the infection through contaminated food while
attending a conference for newly ordained bishops in Italy last month.
Symptoms of hepatitis A include fever, tiredness, loss of appetite,
nausea and abdominal discomfort. Fargo Catholic Diocese in North
Dakota (pictured) is where the bishop is located.<EOS><pad>BishopJohn
Folda, of North Dakota, is taking time off after being diagnosed. He
contracted the infection through contaminated food in Italy. Church
members in Fargo, Grand Forks and Jamestown could have been
exposed.<EOS>


Out[8] - (1, 1200)

Out[9] - 

[   27 23176  4694  1779  1343    28   506  1091   132    28   570     6
    78  7124   192 14454    15  3570  2067    23    46 26133    17  1019
   635    91     3  5349 23421   494     6 10487     2   728     2  1353
  3156   278  1838    28   736   809    28 13481  7511    22   625    28
  1311  2396     3   187    22  1353  1510   181 16146  1049   320   103
     2    22 26563   651   467   213   826   192  3156  1262    28 13131
     4   186 16949    17    71 12319  6604   828 29725     4     5  1081
  1083   213    54   138     3  5349 23421   494     6 10487     2   728
     8   346    12  1353   354    15  3570  2067  7511    22 24497   570
     6    78    71   213  1081   144  3360   691 12319  6604   828     2
   705     8   231    24   305   710   272  1838    68  6341     3     9
   570     6    78  7124   436   219   132   560   429     3   368 23421
   494     6 10487     7     5  1081  1353 10874 20919   217     8 12370
    21    12  2713   127 23421   494     6 10487    40 23176   809   518
   150   181   290  3892   275   527  8947   171  1269   936   213  9025
     3    69  1353   233  8272   527  6056   583   691  4398  3156   809
 14507  5429   812  7356     3  3622  6604   828     2    28   705     6
   104     6   292 15004   181 29725     4     5 21961  1838 10687    45
     2 11985   527 11907  5364     2    40    43  1383   213  2801  1248
  1078   809    28 13481    35    40    19 23176   116  4016     2   864
   127     3   305  1353  3156 17775 12979  3095   186    77  1353   669
 27439  6050 13459  1628  1290   131   143    18   757   320  2501   213
 25725 29725     2    41   969     3 16978  1822  9855  1962     2 17347
    16     2   127  4601 27439  6050 13459  1628  5349 23421   494     6
 10487 29725     4     5  3156  2868   132   213 15191   583   527    28
   506  1091     2 12319  6604   828     2    28   583   285   143    18
    46 13488 23707  6050 13459  1628   368 23421   494     6 10487   436
   213   884   320  3429    61    15  3570  2067  6715  3156   186     2
   673  1510   181 16146  1049   320   824  1311  2396     2  1353    90
 15438    17   285    22  2214   320 17950    28   346     6   650 13131
     4     2  7228   213  1052   763   314    71   213  2358   527  3622
  6604   828 29725     4     5 18352  2398  1081     3  3622  6604   828
  1353  7214   213 19839   277   527    68 27439  9275  1628 12320  5403
  9242  5590  2385    35   710   272  1838    68  6341   132  2642 23707
  6050 13459  1628  3622  6604   828   669 27884     4    40 27872   391
    28  5302   531  2504   527    68     3   305  1353    43  4925   278
   523  1383   163 20812  2801  1248  1078   186  1353  3156 17775 12979
  3095 23707  6050 13459  1628   305    40  5945   320  1242    68  1078
  7511   131   540   278   320  8916   285   131    40  2362 15627     3
  1561  1078  8075   114   369  1613  1838    68   102    41  7584    17
   458 23707  6050 13459  1628  3622  6604   828 29725     4     5   583
   132    97  2861  6107 17946     5   213  6349   527   354    28   650
     6   475  3570  2067  6715  3156  4172 29725   391  2713    25  3630
   320   245 17388   181  1884  4140  1838 23421   494     6 10487  1820
     2    35   132  4140   329   926   102   213  5556    22  1353    86
 25070   918   155   213  6700     6  2057  3602     3     9  4038  2256
  1248   864   285    22    62    18    46    95   213  3602   809   213
    55    15   651  6866  4604   279  1205  3622  6604   828 29725     4
     5  2498 12320  5403  9242  5590  2385    78    28   826   542 15902
  3569     2 11985   527 11907  5364     2    78   560   253     2   429
     3   405  2067   992  1606    22  1353    43 17997   595   239   213
    55   527   213  7124     3  6753  1565  8120   479     2  1838 12887
 26509 21380   328 29725     4     5  1839 25725  2694  1676     2   127
  3611   871  5784  1435  1248 12319     7     5   228   809   824    55
     3   305    40    46    64  1248  1078   809    28 13481   132 15010
  7301   285  2801     2    35    40    19    40   116  4016  1782   871
  2694  1606   285    77  1353  1290   131   143    18   757   320  2501
   213 25725   186  8075   114   103   919    68    68   177  1782   368
 23421   494     6 10487    40   346   126   132 15902  3569   186  1326
  1248  1078   809    28 13481  4872    22  6005  6929   809   518   150
   320   290  3892   275   527  7468    81     3    69 12402     7    26
   209   346   213 13481   320   955   278  7511   213 25725  1841   809
   239   128    10  3229  2535  1782   129  8198     7    26   217   320
   245 17388   181  1884  4140  1838   134  1820   186   849  1884   576
   329   926   102   213 25725  1606    22  1353 25070   918   155   213
  3602     2    51  2253    22    62    18    46    95   213  3602   809
   213    55   527   213 25725   186   132 13040  2398    61   592     2
   213  4038  2256  1782     9   641   527    15  2067   992  1606   285
    22  1353 17997   595    78    15  2067   239   213    55   527   213
 25725    90   103     7     5  1232   761   824    62    43    18  3625
   320    15  4398  3156   186  1201   527   490  2002 23421   494     6
 10487  1353   233  8272   527  6056   583   691  4398  3156   355    28
  2145   809 14507  5429   812     8 12370    21    12    69   969  3611
   368 23421   494     6 10487    39   169  3263   635    91   936  5892
     2    35 12319     7     5   228    18   913    68  8232  1782    13
  1525   824    39   191   101   362  3060   171  6642   116  4016   186
  1269   936   213  9025     2   181   354    28  2067   640    41     7
   165    78   213   826  1782     9 26024   527  6700  3156   186  3156
  6715   354    28  3570  2067  1435  3787     3  2994  1779   952   320
   124    90   993  3736    28  3537    55   132  2173     3    56   347
  6335   141  7270 15191   213  4472   527 16972   595    97 23891  6412
    49  1151 20327 27439  6050 13459  1628   368 23421   494     6 10487
    39   169  3263   635    91   936  5892     2    35 12319 29725     4
     5   228    18   913    68  1019   545     3    13  1525   824    39
   191   101   362  3060   171  6642   116  4016   186  1269   936   213
  9025     2   181   354    28  2067   640    41 29725     4   165    78
   213   826     3    56   347  6335   141  7270 15191   213  4472   527
 16972   595    97 23891  6412    49  1151  4172 29725   391 23421   494
     6 10487     2   527 14735     2 11985   527 11907  5364     2  1353
    43 24306  5831  4461  1838  3156  1019  1223    91 27439  9275  1628
   102  1480    22    39    18   320   976   163  2008   165     6  1166
    10     1     0  5349 23421   494     6 10487     2   728     2    40
 23176   809   518   150  3892   275   171  3156  1081  4172 27439  6774
  1628  5670   354  2067  7511    22 26563   651   467   826   132 15902
  3569     2 11985   527 11907  5364  4172 27439  6774  1628  3481  3094
   570     6    78    71   705     6   104     6   292 12319  6604   828
     7     5  1081     2  1779   710   132  2642  4172 27439  6774  1628
  2713   476    22    62    18    46    95   904  6700     6  2057  3602
   809    55   527  7124  4172 27439  6774  1628    69  1353   233  8272
   809 14507  5429   812   527  6056   583   691  4398  3156    10     1]



Out[10] - 

Article:

 A drunk driver who killed a young woman in a head-on crash while
checking his mobile phone has been jailed for six years. Craig
Eccleston-Todd, 27, was driving home from a night at a pub when he
received a text message. As he was reading or replying to it, he
veered across the road while driving round a bend and smashed into
Rachel Titley’s car coming the other way. Craig Eccleston-Todd, 27
(left) was using his mobile phone when he crashed head-on into the car
being driven by Rachel Titley, 28 (right). She died later from her
injuries. The head-on crash took place in October 2013. Mr Eccleston-
Todd's car was barely recognisable (pictured) Police said Eccleston-
Todd had drunk at least three or four pints of beer before getting
behind the wheel. He was found guilty of causing death by dangerous
driving at Portsmouth Crown Court yesterday. Miss Titley, a 28-year-
old solicitor’s clerk from Cowes, Isle of Wight, had also spent the
evening with friends at a pub but had not drunk any alcohol, police
said. She was driving responsibly and there was ‘nothing she could
have done to avoid the collision’, they added. Lindsay Pennell,
prosecuting, said: ‘Craig Eccleston-Todd’s driving resulted in the
tragic death of a young woman, Rachel Titley, a death that could have
been avoided. ‘Mr Eccleston-Todd took the decision to pick up his
mobile phone whilst driving and, either reading or replying to this
text message, was so distracted that he failed to negotiate a left-
hand bend, crossing the central white line into the path of Miss
Titley’s oncoming car. Miss Titley was pulled the wreckage of
her Daihatsu Cuore but died later from her injuries in hospital. ‘Miss
Titley [had] a bright future ahead of her. She was also returning home
having spent an enjoyable evening with friends and was driving
responsibly. ‘She had arranged to contact her friends when she got
home to confirm that she had arrived safely. Her friends sadly never
heard from her after they parted company. ‘Miss Titley’s death in
these circumstances reiterates the danger of using a hand-held mobile
phone whilst driving.’ Police were unable to take breath or blood
tests from Eccleston-Todd immediately, but in tests several hours
after the accident he was only marginally under the drink-drive limit.
The judge agreed with police that he would have been over the limit at
the time his red Citroen hit Miss Titley’s blue Daihatsu Cuore on a
road near Yarmouth, Isle of Wight, on October 11, 2013. His phone
records showed he was also texting around the time of the crash. PC
Mark Furse, from Hampshire constabulary’s serious collision
investigation unit, said: 'Our thoughts are with Rachel's family at
this time. She had been out with friends at a pub in Shalfleet that
evening, but had not had any alcohol. 'Our investigation showed that
there was nothing she could have done to avoid the collision and sadly
it cost her her life. 'Mr Eccleston-Todd had left work in Yarmouth and
met with friends at a pub where he drank at least three to four pints
of lager. He hadn't long left the pub to return home when the
collision occurred at around 9.30pm. 'We weren't able to take breath
or blood tests from him immediately and although blood taken several
hours after the collision showed he was marginally under the limit, we
maintain he would have been over the limit at the time of the
collision and in summing up today, the judge agreed. 'The analysis of
his phone records showed that he was texting on his phone around the
time of the collision so it's highly likely this would also have
contributed to his dangerous driving and loss of control.' Eccleston-
Todd was found guilty of causing death by dangerous driving following
a trial at Portsmouth Crown Court (pictured) He added: 'Mr Eccleston-
Todd will now spend six years behind bars, but Rachel's family have
lost her forever. 'I hope this will make people think twice before
drinking any alcohol and getting behind the wheel, or using a phone
once they're on the road. 'The dangers of drink driving and driving
whilst using a mobile phone are obvious. Those who continue to do so
risk spending a substantial time in prison. This case highlights just
how tragic the consequences of committing these offences can be.' ‘Mr
Eccleston-Todd will now spend six years behind bars, but Rachel’s
family have lost her for ever. I hope this will make people think
twice before drinking any alcohol and getting behind the wheel, or
using a phone once they’re on the road. This case highlights just how
tragic the consequences of committing these offences can be.’
Eccleston-Todd, of Newport, Isle of Wight, was also disqualified from
driving for eight years after which he will have to complete an
extended re-test.<EOS><pad>CraigEccleston-Todd, 27, had drunk at least
three pints before driving car. Was using phone when he veered across
road in Yarmouth, Isle of Wight. Crashed head-on into 28-year-old
Rachel Titley's car, who died in hospital. Police say he would have
been over legal drink-drive limit at time of crash. He was found
guilty at Portsmouth Crown Court of causing death by dangerous
driving.<EOS>

Out[12] - 

query shape: (2, 3)

[[1 0 0]
 [0 1 0]]

key shape: (2, 3)

[[1 2 3]
 [4 5 6]]

value shape: (2, 3)

[[0 1 0]
 [1 0 1]]

mask shape: (2, 2)

[[ 0.e+00  0.e+00]
 [-1.e+09  0.e+00]]

Expected Output:

query shape: (2, 3)

[[1 0 0]
 [0 1 0]]

key shape: (2, 3)

[[1 2 3]
 [4 5 6]]

value shape: (2, 3)

[[0 1 0]
 [1 0 1]]

mask shape: (2, 2)

[[ 0.e+00  0.e+00]
 [-1.e+09  0.e+00]]

Out[13] -

query dot key shape: (2, 2)

[[0.57735026 2.309401  ]
 [1.1547005  2.8867514 ]]

Expected Output:

query dot key shape: (2, 2)

[[0.57735026 2.309401  ]
 [1.1547005  2.8867514 ]]

Out[14] - 

masked query dot key shape: (2, 2)

[[ 5.7735026e-01  2.3094010e+00]
 [-1.0000000e+09  2.8867514e+00]]

Expected Output:

masked query dot key shape: (2, 2)

[[ 5.7735026e-01  2.3094010e+00]
 [-1.0000000e+09  2.8867514e+00]] 

Out[15] - 
masked query dot key dot value shape: (2, 3)

[[ 2.3094010e+00  5.7735026e-01  2.3094010e+00]
 [ 2.8867514e+00 -1.0000000e+09  2.8867514e+00]]

Expected Output:

masked query dot key dot value shape: (2, 3)

[[ 2.3094010e+00  5.7735026e-01  2.3094010e+00]
 [ 2.8867514e+00 -1.0000000e+09  2.8867514e+00]]

Out[16] - 

uery with batch dim shape: (1, 2, 3)

[[[1 0 0]
  [0 1 0]]]

key with batch dim shape: (1, 2, 3)

[[[1 2 3]
  [4 5 6]]]

value with batch dim shape: (1, 2, 3)

[[[0 1 0]
  [1 0 1]]]

boolean mask shape: (2, 2)

[[ True  True]
 [False  True]]

Expected Output:

query with batch dim shape: (1, 2, 3)

[[[1 0 0]
  [0 1 0]]]

key with batch dim shape: (1, 2, 3)

[[[1 2 3]
  [4 5 6]]]

value with batch dim shape: (1, 2, 3)

[[[0 1 0]
  [1 0 1]]]

boolean mask shape: (2, 2)

[[ True  True]
 [False  True]]

Out[18] - 

Array([[[0.8496746 , 0.15032545, 0.8496746 ],
        [1.        , 0.        , 1.        ]]], dtype=float32)
Expected Output:

DeviceArray([[[0.8496746 , 0.15032545, 0.8496746 ],
              [1.        , 0.        , 1.        ]]], dtype=float32)

Out[19] - All tests passed 

Out[49] - All tests passed

<br/><br/>
</pre>
