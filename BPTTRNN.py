# https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/4https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/4
import torch
from torch import nn
import torch.nn.functional as F
# 
# seq_len = 20
# layer_size = 50
# idx = 0

class TBPTT():
    def __init__(self, one_step_module, loss_module, k1, k2, optimizer):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2
        # You can also remove all the optimizer code here, and the
        # train function will just accumulate all the gradients in
        # one_step_module parameters
        self.optimizer = optimizer

    def train(self, input_sequence, init_state):
        states = [(None, init_state)]
        for j, (inp, target) in enumerate(input_sequence):

            state = states[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state)
            states.append((state, new_state))

            while len(states) > self.k2:
                # Delete stuff that is too old
                del states[0]

            if (j+1)%self.k1 == 0:
                loss = self.loss_module(output, target)

                self.optimizer.zero_grad()
                # backprop last module (keep graph only if they ever overlap)
                start = time.time()
                loss.backward(retain_graph=self.retain_graph)
                for i in range(self.k2-1):
                    # if we get all the way back to the "init_state", stop
                    if states[-i-2][0] is None:
                        break
                    curr_grad = states[-i-1][0].grad
                    states[-i-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                print("bw: {}".format(time.time()-start))
                self.optimizer.step()





class MyMod(nn.Module):
    def __init__(self):
        super(MyMod, self).__init__()
        self.lin = nn.Linear(2*layer_size, 2*layer_size)

    def forward(self, inp, state):
        global idx
        full_out = self.lin(torch.cat([inp, state], 1))
        # out, new_state = full_out.chunk(2, dim=1)
        out = full_out.narrow(1, 0, layer_size)  # 0:layer_size 
        new_state = full_out.narrow(1, layer_size, layer_size)  # layer_size:(2 * layersize)
        def get_pr(idx_val):
            def pr(*args):
                print("doing backward {}".format(idx_val))
            return pr
        new_state.register_hook(get_pr(idx))
        out.register_hook(get_pr(idx))
        print("doing fw {}".format(idx))
        idx += 1
        return out, new_state


# def bptt(self, x, y):
#     T = len(y)
    
#     # Perform forward propagation
#     o, s = self.forward_propagation(x)
    
#     # We accumulate the gradients in these variables
#     dLdU = np.zeros(self.U.shape)
#     dLdV = np.zeros(self.V.shape)
#     dLdW = np.zeros(self.W.shape)
#     delta_o = o
#     delta_o[np.arange(len(y)), y] -= 1.0
    
#     # For each output backwards…
#     for t in np.arange(T)[::-1]:
#         dLdV += np.outer(delta_o[t], s[t].T)
        
#         # Initial delta calculation: dL/dz
#         delta_t = self.V.T.dot(delta_o[t]) * (1 – (s[t] ** 2))
        
#         # Backpropagation through time (for at most self.bptt_truncate steps)
#         for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
#             # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            
#             # Add to gradients at each previous step
#             dLdW += np.outer(delta_t, s[bptt_step-1])
#             dLdU[:,x[bptt_step]] += delta_t
            
#             # Update delta for next step dL/dz at t-1
#             delta_t = self.W.T.dot(delta_t) * (1 – s[bptt_step-1] ** 2)
        
#     return [dLdU, dLdV, dLdW]