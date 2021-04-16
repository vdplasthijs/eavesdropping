relu folder means that the predictin output nonlinearity = softmax(relu(x))

This was done to unify with G1 G2 output.

But I realised that G1 G2 need the relu to prevent w_G1 = - w_G2, not needed for prediction so why not leave out 

Results were similar (combined task learns, dmc only doesnt), but learning is slower
