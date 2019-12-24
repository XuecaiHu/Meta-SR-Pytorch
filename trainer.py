import os
import math
from decimal import Decimal

import utility
import pdb
import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8


    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network

    def input_matrix_wpn_new(self, inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale * inH), int(scale * inW)
        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH, scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)


        ####projection  coordinate  and caculate the offset
        h_project_coord = torch.arange(0, outH, 1).mul(1.0 / scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).mul(1.0 / scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag, 0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)


        mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
        mask_mat = mask_mat.eq(2)

        i = 1
        h, w,_ = pos_mat.size()
        while(pos_mat[i][0][0]<= 1e-6 and i<h):
            i = i+1

        j = 1
        #pdb.set_trace()
        h, w,_ = pos_mat.size()
        while(pos_mat[0][j][1]<= 1e-6 and j<w):
            j = j+1

        pos_mat_small = pos_mat[0:i,0:j,:]

        pos_mat_small = pos_mat_small.contiguous().view(1, -1, 2)
        if add_scale:
            scale_mat = torch.zeros(1, 1)
            scale_mat[0, 0] = 1.0 / scale
            scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(1)), 0)  ###(inH*inW*scale_int**2, 4)
            pos_mat_small = torch.cat((scale_mat.view(1, -1, 1), pos_mat_small), 2)

        return pos_mat_small, mask_mat  ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

        ########speed up the model by removing the computation

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            N,C,H,W = lr.size()
            _,_,outH,outW = hr.size()
            scale_coord_map, mask = self.input_matrix_wpn_new(H,W,self.args.scale[idx_scale])  ###  get the position matrix, mask

            if self.args.n_GPUs>1 and not self.args.cpu:
                scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
            else:
                scale_coord_map = scale_coord_map.to(device)
            
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale, scale_coord_map)
            re_sr = torch.masked_select(sr,mask.to(device))
            re_sr = re_sr.contiguous().view(N,C,outH,outW)
            loss = self.loss(re_sr, hr)
            
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if self.args.n_GPUs == 1:
            target = self.model
        else:
            target = self.model  #.module

        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
        )
        ## save models

    def test(self):  
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_acc_ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                #tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    N,C,H,W = lr.size()
                    scale = self.args.scale[idx_scale]
                    outH,outW = int(H*scale),int(W*scale)
                    #_,_,outH,outW = hr.size()
                    #timer_test.tic()

                    scale_coord_map, mask = self.input_matrix_wpn_new(H,W,self.args.scale[idx_scale])
                    #position, mask = self.pos_matrix(H,W,self.args.scale[idx_scale])
                    #print(timer_test.toc())
                    if self.args.n_GPUs>1 and not self.args.cpu:
                        scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
                    else:
                        scale_coord_map = scale_coord_map.to(device)

                    timer_test.tic()
                    sr = self.model(lr, idx_scale,scale_coord_map)
                    timer_test.hold()
                    re_sr = torch.masked_select(sr,mask.to(device))
                    sr = re_sr.contiguous().view(N,C,outH,outW)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    #timer_test.hold()
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        a=1
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
               # print(timer_test.acc/100)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        eval_acc_ssim / len(self.loader_test),
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

