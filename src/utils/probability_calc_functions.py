    
    def _min_dist_NN_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # extract num_NN nearest neighbours
        ranks = ranks[:num_NN]

        # Extract means, variance, and classes
        means_NN = self.means[:,ranks].cuda()
        vars_NN = self.vars[ranks].cuda()
        classes_NN = [self.classes[i] for i in ranks]
    
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN*num_MC).reshape(num_NN,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        
        # Find smallest distance to image
        _, indx_min = torch.min(dist_to_NN,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts.cpu()/num_MC
        probs = {key:0 for key in np.unique(self.classes)}
        for i in range(len(counts)):
            probs[classes_NN[indx_class[i].item()]]+=counts[i].item()
        
        return probs
    
    def _min_dist_NN_with_min_dist_rand_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # extract num_NN nearest neighbours
        ranks = ranks[:num_NN]

        # Extract means, variance, and classes
        means_NN = self.means[:,ranks].cuda()
        vars_NN = self.vars[ranks].cuda()
        classes_NN = [self.classes[i] for i in ranks]
    
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN*num_MC).reshape(num_NN,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        
        # Find smallest distance to image
        _, indx_min = torch.min(dist_to_NN,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts.cpu()/num_MC
        probs = {key:0 for key in np.unique(self.classes)}
        for i in range(len(counts)):
            probs[classes_NN[indx_class[i].item()]]+=counts[i].item()
        
        
        # Extract as many objects from each class as the one with the lowest count
        min_samples_class = min(self.num_samples_classes.values())
        ranks_new = np.zeros((self.num_classes*min_samples_class,))
        ranks_dict = {}
        for i, class_ in enumerate(self.num_samples_classes.keys()):
            class_selections = np.random.choice(self.classes_idxs[class_],
                                                      min_samples_class, False)
            ranks_new[i*min_samples_class:(i+1)*min_samples_class] = class_selections
            ranks_dict[class_] = class_selections

        # Extract means, variance, and classes
        means_NN = self.means[:,ranks_new].cuda()
        vars_NN = self.vars[ranks_new].cuda()
        
        # Parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        df = len(mean_emb)
        
        # Sample dist
        num_each_class = 15
        dist_to_NN = torch.zeros((num_MC,num_each_class*self.num_classes)).cuda()
        for i in range(num_MC):
            idxs_c = np.random.random_integers(0,min_samples_class-1,num_each_class)
            idxs = []
            [idxs.extend(list(j*min_samples_class+idxs_c)) for j in range(self.num_classes)]
            dist_to_NN[i,:] = (scaling[idxs]*torch.Tensor(noncentral_chisquare(df,nonc[idxs],)).cuda())
        
        # Accumulate on class level per sample
        idx = []
        [idx.extend([i]*num_each_class) for i in range(self.num_classes)]

        # Accumulate on class level per sample
        idx = torch.Tensor(idx).type(torch.int64).cuda()
        dist_to_NN_class = torch.zeros((num_MC,len(self.unique_classes))).cuda().T
        scatter_mean(src=dist_to_NN.T,index=idx,out=dist_to_NN_class,dim=0)
        
        # Find closest class per sample
        indx_min = torch.argmin(dist_to_NN_class,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts/num_MC
        probs2 = {key:0 for key in self.unique_classes}
        for i in range(len(counts)):
            probs2[self.unique_classes[indx_class[i].item()]]+=counts[i].item()
        
        for class_ in self.unique_classes:
            probs[class_]=(probs[class_]*3+probs2[class_])/4
        
        return probs
    
    def _avg_dist_rand_(self, mean_emb, var_emb, num_NN, num_MC):
        # Extract as many objects from each class as the one with the lowest count
        min_samples_class = min(self.num_samples_classes.values())
        ranks_new = np.zeros((self.num_classes*min_samples_class,))
        ranks_dict = {}
        for i, class_ in enumerate(self.num_samples_classes.keys()):
            class_selections = np.random.choice(self.classes_idxs[class_],
                                                      min_samples_class, False)
            ranks_new[i*min_samples_class:(i+1)*min_samples_class] = class_selections
            ranks_dict[class_] = class_selections

        # Extract means, variance, and classes
        means_NN = self.means[:,ranks_new].cuda()
        vars_NN = self.vars[ranks_new].cuda()
        
        # Parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        df = len(mean_emb)
        
        # Sample dist
        num_each_class = num_NN# int(num_NN/self.num_classes)*
        dist_to_NN = torch.zeros((num_MC,num_each_class*self.num_classes)).cuda()
        for i in range(num_MC):
            idxs_c = np.random.random_integers(0,min_samples_class-1,num_each_class)
            idxs = []
            [idxs.extend(list(j*min_samples_class+idxs_c)) for j in range(self.num_classes)]
            dist_to_NN[i,:] = (scaling[idxs]*torch.Tensor(noncentral_chisquare(df,nonc[idxs],)).cuda())
        
        # Accumulate on class level per sample
        idx = []
        [idx.extend([i]*num_each_class) for i in range(self.num_classes)]

        # Accumulate on class level per sample
        idx = torch.Tensor(idx).type(torch.int64).cuda()
        dist_to_NN_class = torch.zeros((num_MC,len(self.unique_classes))).cuda().T
        scatter_mean(src=dist_to_NN.T,index=idx,out=dist_to_NN_class,dim=0)
        
        # Find closest class per sample
        indx_min = torch.argmin(dist_to_NN_class,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts/num_MC
        probs = {key:0 for key in self.unique_classes}
        for i in range(len(counts)):
            probs[self.unique_classes[indx_class[i].item()]]+=counts[i].item()

        return probs
    
    def _min_dist_NN_cap_class_NN_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # extract num_NN nearest neighbours
        ranks = ranks[:num_NN]

        # Extract classes
        classes_NN = [self.classes[i] for i in ranks]
        
        # Remove classes filling up more than 1/4 of num_NN
        class_count = {self.unique_classes[i]:0 for i in range(len(self.unique_classes))}
        ranks_list = []
        for i in range(num_NN):
            if class_count[classes_NN[i]] < num_NN*1/4:
                class_count[classes_NN[i]] += 1
                ranks_list.append(ranks[i].item())
            
        ranks = torch.Tensor(ranks_list).type(torch.int64)
        num_NN = len(ranks)
        # Extract means, variance, and classes
        means_NN = self.means[:,ranks].cuda()
        vars_NN = self.vars[ranks].cuda()
        classes_NN = [self.classes[i] for i in ranks]
    
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN*num_MC).reshape(num_NN,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        
        # Find smallest distance to image
        _, indx_min = torch.min(dist_to_NN,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts.cpu()/num_MC
        probs = {key:0 for key in np.unique(self.classes)}
        for i in range(len(counts)):
            probs[classes_NN[indx_class[i].item()]]+=counts[i].item()
            
        return probs
    
    def _avg_dist_NN_cap_class_NN_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # extract num_NN nearest neighbours
        ranks = ranks[:num_NN]

        # Extract classes
        classes_NN = [self.classes[i] for i in ranks]
        
        # Remove classes filling up more than 1/4 of num_NN
        class_count = {self.unique_classes[i]:0 for i in range(len(self.unique_classes))}
        ranks_list = []
        for i in range(num_NN):
            if class_count[classes_NN[i]] < num_NN*1/4:
                class_count[classes_NN[i]] += 1
                ranks_list.append(ranks[i].item())
            
        ranks = torch.Tensor(ranks_list).type(torch.int64)
        num_NN = len(ranks)
        
        # Extract means, variance, and classes
        means_NN = self.means[:,ranks].cuda()
        vars_NN = self.vars[ranks].cuda()
        classes_NN = [self.classes[i] for i in ranks]
    
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + vars_NN
        delta = (mean_emb - means_NN.T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN*num_MC).reshape(num_NN,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        
        # Accumulate on class level per sample
        _, idx = np.unique(classes_NN, return_inverse=True)
        idx = torch.Tensor(idx).type(torch.int64).cuda()
        dist_to_NN_class = torch.zeros((num_MC,len(self.unique_classes))).cuda().T
        scatter_mean(src=dist_to_NN,index=idx,out=dist_to_NN_class,dim=0)
        
        # Find closest class per sample
        indx_min = torch.argmin(dist_to_NN_class,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts/num_MC
        probs = {key:0 for key in self.unique_classes}
        for i in range(len(counts)):
            probs[classes_NN[indx_class[i].item()]]+=counts[i].item()

        return probs
            
                
    def _min_dist_NN_pr_class_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # Init dropout dist
        dropout_dist = torch.distributions.Bernoulli(torch.tensor([0.25]))
        num_NN_classes = len(self.unique_classes)*num_NN
        
        # extract num_NN nearest neighbours from each class
        class_count = {self.unique_classes[i]:0 for i in range(len(self.unique_classes))}
        ranks_list = []
        for i in range(len(ranks)):
            rank_class = self.classes[ranks[i].item()]
            if class_count[rank_class] < num_NN:
                class_count[rank_class] += 1
                ranks_list.append(ranks[i].item())
            if sum(class_count.values()) == num_NN_classes:
                break
    
        ranks = torch.tensor(ranks_list)
        classes = np.array(self.classes)[ranks]
        
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + self.vars[ranks]
        delta = (mean_emb - self.means[:,ranks].T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        df = len(mean_emb)
        
        # Start sampling
        dist_to_NN = torch.zeros(num_MC,num_NN_classes)
        for i in range(num_NN_classes):             
            # Simulate distances for all NN
            dist_to_NN[:,i] = scaling[i].cpu()*torch.Tensor(noncentral_chisquare(df,nonc[i],num_MC))
            dropout = dropout_dist.sample((num_MC,)).T[0]
            dist_to_NN[:,i] = dist_to_NN[:,i]+1e5*dropout
        
        
        #class_u, idx = np.unique(classes, return_inverse=True)
        #(torch.bincount(torch.tensor(idx).cuda(), self.vars[ranks].cuda())/num_NN)
        # Find smallest distance to image
        _, indx_min = torch.min(dist_to_NN,1)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts/num_MC
        probs = {key:0 for key in self.unique_classes}
        for i in range(len(counts)):
            probs[classes[indx_class[i].item()]]+=counts[i].item()

        return probs
    
    
    def _avg_dist_NN_pr_class_(self, ranks, mean_emb, var_emb, num_NN, num_MC):
        # Init dropout dist
        num_NN_classes = len(self.unique_classes)*num_NN
        
        # extract num_NN nearest neighbours from each class
        class_count = {self.unique_classes[i]:0 for i in range(len(self.unique_classes))}
        ranks_list = torch.zeros((num_NN_classes,),dtype=torch.int64)
        count = 0
        #pdb.set_trace()
        for i in range(len(ranks)):
            rank_class = self.classes[ranks[i].item()]
            if class_count[rank_class] < num_NN:
                class_count[rank_class] += 1
                ranks_list[count] = ranks[i].item()
                count += 1
                
                if count == num_NN_classes:
                    break
        
        #ranks_list = np.repeat([0],540)
        classes = np.array(self.classes)[ranks_list]
        
        # Calc scaled non-centered chi-sq dists parameters
        scaling = var_emb + self.vars[ranks_list]
        delta = (mean_emb - self.means[:,ranks_list].T).T
        nonc = (scaling**(-1)*torch.diag(torch.matmul(delta.T,delta))).cpu().numpy()
        nonc = np.repeat(nonc,num_MC).reshape(num_NN_classes,num_MC).T
        df = len(mean_emb)
        df = np.repeat(df,num_NN_classes*num_MC).reshape(num_NN_classes,num_MC).T
        
        # Sample dist
        dist_to_NN = (scaling*torch.Tensor(noncentral_chisquare(df,nonc,)).cuda()).T
        
        # Accumulate on class level per sample
        _, idx = np.unique(classes, return_inverse=True)
        idx = torch.Tensor(idx).type(torch.int64).cuda()
        dist_to_NN_class = torch.zeros((num_MC,len(self.unique_classes))).cuda().T
        scatter_mean(src=dist_to_NN,index=idx,out=dist_to_NN_class,dim=0)
        
        # Find closest class per sample
        indx_min = torch.argmin(dist_to_NN_class,0)
        indx_class, counts = indx_min.unique(return_counts=True)
        counts = counts/num_MC
        probs = {key:0 for key in self.unique_classes}
        for i in range(len(counts)):
            probs[classes[indx_class[i].item()]]+=counts[i].item()

        return probs
