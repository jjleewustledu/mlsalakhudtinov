classdef DBM 
	%% DBM is a derivative work arising from 
    % Version 1.000 Code provided by Ruslan Salakhutdinov.
    % http://www.cs.cmu.edu/~rsalakhu/index.html
    %
    % Permission is granted for anyone to copy, use, modify, or distribute this
    % program and accompanying programs and documents for any purpose, provided
    % this copyright notice is retained and prominently displayed, along with
    % a note saying that the original programs are available from our
    % web page.
    %
    % The programs and documents are distributed without any warranty, express or
    % implied.  As the programs were written for research purposes only, they have
    % not been tested to the degree that would be advisable in any important
    % application.  All use of these programs is entirely at the user's own risk.

	%  $Revision$
 	%  was created 01-Oct-2017 23:51:02 by jjlee,
 	%  last modified $LastChangedDate$ and placed into repository /Users/jjlee/Local/src/mlcvl/mlsalakhudtinov/src/+mlsalakhudtinov.
 	%% It was developed on Matlab 9.3.0.713579 (R2017b) for MACI64.  Copyright 2017 John Joowon Lee.
 	
	properties
        CD
 		meanFieldUpdates = 10 % Number of the mean-field updates.  I also used 30 MF updates.
        maxepoch % maximum number of epochs
        numbatches
        numcases
        numdim
        numdims
        numhid % number of hidden units
        numhids
        numpen
        numpens
        batchdata % the data that is divided into batches (numcases numdims numbatches)
        batchtargets
        restart = 1 % set to 1 if learning starts from beginning
        testbatchdata
        testbatchtargets
        
        poshidprobs
        neghidprobs
        posprods
        negprods
 	end

	methods 
        
        %%
		  
        function this = demo_small(this)
            %% DEMO_SMALL.m Main file for training and fine-tuning a toy DBM model.
            randn('state',100);
            rand( 'state',100);
            warning off
            
            %clear all
            close all
            
            fprintf(1,'Converting Raw files into Matlab format \n');
            this = this.converter;
            
            fprintf(1,'Pretraining a Deep Boltzmann Machine. \n');
            this = this.makebatches;
            [this.numcases,this.numdims,this.numbatches] = size(this.batchdata);
            
            %%%%%% Training 1st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            this.numhid   = 10; 
            this.maxepoch = 10;
            fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',this.numdims,this.numhid);
            this.restart  = 1;
            this = this.rbm;
            
            %%%%%% Training 2st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            close all
            this.numpen   = 10;
            this.maxepoch = 10;
            fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',this.numhid,this.numpen);
            this.restart  = 1;
            this = this.makebatches;
            this = this.rbm_l2;
            
            
            %%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            close all
            this.numhid   = 10;
            this.numpen   = 10;
            this.maxepoch = 10;
            
            fprintf(1,'Learning a Deep Bolztamnn Machine. \n');
            this.restart = 1;
            this = this.makebatches;
            this = this.dbm_mf;
            
            %%%%%% Fine-tuning two-layer Boltzmann machine  for classification %%%%%%%%%%%%%%%%%
            this.maxepoch = 10;
            this = this.makebatches;
            this = this.backprop;
        end
        function this = demo(this)
            %% DEMO.m Main file for training and fine-tuning a DBM model (reproduces results of the DBM paper).
            
            randn('state',100);
            rand( 'state',100);
            warning off
            
            %clear all
            close all
            
            fprintf(1,'Converting Raw files into Matlab format \n');
            this = this.converter;
            
            fprintf(1,'Pretraining a Deep Boltzmann Machine. \n');
            this = this.makebatches;
            [this.numcases,this.numdims,this.numbatches] = size(this.batchdata);
            
            %%%%%% Training 1st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            this.numhid   = 500; 
            this.maxepoch = 100;
            fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',this.numdims,this.numhid);
            this.restart  = 1;
            this = this.rbm;
            
            %%%%%% Training 2st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            close all
            this.numpen   = 1000;
            this.maxepoch = 200;
            fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,this.numpen);
            this.restart  = 1;
            this = this.makebatches;
            this = this.rbm_l2;            
            
            %%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            close all
            this.numhid   = 500;
            this.numpen   = 1000;
            this.maxepoch = 300; % To get results in the paper I used maxepoch=500, which took over 2 days or so.
            
            fprintf(1,'Learning a Deep Bolztamnn Machine. \n');
            this.restart = 1;
            this = this.makebatches;
            this = this.dbm_mf;
            
            %%%%%% Fine-tuning two-layer Boltzmann machine for classification %%%%%%%%%%%%%%%%%
            this.maxepoch = 100;
            this = this.makebatches;
            this = this.backprop;
        end
        function this = converter(this)
            %% CONVERTER.m Converts raw MNIST digits into matlab format
            
            % This program reads raw MNIST files available at
            % http://yann.lecun.com/exdb/mnist/
            % and converts them to files in matlab format
            % Before using this program you first need to download files:
            % train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
            % t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
            % and gunzip them. You need to allocate some space for this.
            
            % This program was originally written by Yee Whye Teh
            
            if (logical(exist('digit9.mat', 'file')) && logical(exist('test9.mat', 'file')))
                return
            end
            
            % Work with test files first
            fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n');
            
            f = fopen('t10k-images-idx3-ubyte','r');
            [a,count] = fread(f,4,'int32');
            
            g = fopen('t10k-labels-idx1-ubyte','r');
            [l,count] = fread(g,2,'int32');
            
            fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n');
            n = 1000;
            
            Df = cell(1,10);
            for d=0:9
                Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');
            end
            
            for i=1:10
                fprintf('.');
                rawimages = fread(f,28*28*n,'uchar');
                rawlabels = fread(g,n,'uchar');
                rawimages = reshape(rawimages,28*28,n);
                
                for j=1:n
                    fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
                    fprintf(Df{rawlabels(j)+1},'\n');
                end
            end
            
            fprintf(1,'\n');
            for d=0:9
                fclose(Df{d+1});
                D = load(['test' num2str(d) '.ascii'],'-ascii');
                fprintf('%5d Digits of class %d\n',size(D,1),d);
                save(['test' num2str(d) '.mat'],'D','-mat');
            end
            
            
            % Work with trainig files second
            f = fopen('train-images-idx3-ubyte','r');
            [a,count] = fread(f,4,'int32');
            
            g = fopen('train-labels-idx1-ubyte','r');
            [l,count] = fread(g,2,'int32');
            
            fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n');
            n = 1000;
            
            Df = cell(1,10);
            for d=0:9
                Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
            end
            
            for i=1:60
                fprintf('.');
                rawimages = fread(f,28*28*n,'uchar');
                rawlabels = fread(g,n,'uchar');
                rawimages = reshape(rawimages,28*28,n);
                
                for j=1:n
                    fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
                    fprintf(Df{rawlabels(j)+1},'\n');
                end
            end
            
            fprintf(1,'\n');
            for d=0:9
                fclose(Df{d+1});
                D = load(['digit' num2str(d) '.ascii'],'-ascii');
                fprintf('%5d Digits of class %d\n',size(D,1),d);
                save(['digit' num2str(d) '.mat'],'D','-mat');
            end
            
            system('rm *.ascii');
        end
        function this = makebatches(this)
            %% MAKEBATCHES.m Creates minibatches for DBM training
            
            digitdata=[];
            targets=[];
            load digit0; digitdata = [digitdata; D]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];
            load digit1; digitdata = [digitdata; D]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
            load digit2; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
            load digit3; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
            load digit4; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
            load digit5; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
            load digit6; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
            load digit7; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
            load digit8; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
            load digit9; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
            digitdata = digitdata/255;
            
            totnum=size(digitdata,1);
            fprintf(1, 'Size of the training dataset= %5d \n', totnum);
            
            rand('state',0); %so we know the permutation of the training data
            randomorder=randperm(totnum);
            
            this.numbatches=totnum/100;
            this.numdims   =  size(digitdata,2);
            batchsize = 100;
            this.batchdata = zeros(batchsize, this.numdims, this.numbatches);
            this.batchtargets = zeros(batchsize, 10, this.numbatches);
            
            for b=1:this.numbatches
                this.batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
                this.batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
            end
            clear digitdata targets;
            
            digitdata=[];
            targets=[];
            load test0; digitdata = [digitdata; D]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];
            load test1; digitdata = [digitdata; D]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
            load test2; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
            load test3; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
            load test4; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
            load test5; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
            load test6; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
            load test7; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
            load test8; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
            load test9; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
            digitdata = digitdata/255;
            
            totnum=size(digitdata,1);
            fprintf(1, 'Size of the test dataset= %5d \n', totnum);
            
            rand('state',0); %so we know the permutation of the training data
            randomorder=randperm(totnum);
            
            this.numbatches=totnum/100;
            this.numdims  =  size(digitdata,2);
            batchsize = 100;
            this.testbatchdata = zeros(batchsize, this.numdims, this.numbatches);
            this.testbatchtargets = zeros(batchsize, 10, this.numbatches);
            
            for b=1:this.numbatches
                this.testbatchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
                this.testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
            end
            clear digitdata targets;            
            
            %%% Reset random seeds
            rand( 'state',sum(100*clock));
            randn('state',sum(100*clock));
        end
        function this = rbm(this)
            %% RBM.m Training RBM with binary hidden and visible units
            % This program trains Restricted Boltzmann Machine in which
            % visible, binary, stochastic pixels are connected to
            % hidden, binary, stochastic feature detectors using symmetrically
            % weighted connections. Learning is done with 1-step Contrastive Divergence.
            % The program assumes that the following variables are set externally:
            % this.maxepoch  -- maximum number of epochs
            % this.numhid    -- number of hidden units
            % this.batchdata -- the data that is divided into batches (numcases numdims numbatches)
            % this.restart   -- set to 1 if learning starts from beginning
            
            if (this.restart ==1)
                this.restart=0;
                
                epsilonw      = 0.05;   % Learning rate for weights
                epsilonvb     = 0.05;   % Learning rate for biases of visible units
                epsilonhb     = 0.05;   % Learning rate for biases of hidden units
                
                this.CD=1;
                weightcost  = 0.001;
                initialmomentum  = 0.5;
                finalmomentum    = 0.9;
                
                [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
                epoch=1;
                
                % Initializing symmetric weights and biases.
                vishid     = 0.001*randn(this.numdims, this.numhid);
                hidbiases  = zeros(1,this.numhid);
                visbiases  = zeros(1,this.numdims);
                
                this.poshidprobs = zeros(this.numcases,this.numhid);
                this.neghidprobs = zeros(this.numcases,this.numhid);
                this.posprods    = zeros(this.numdims,this.numhid);
                this.negprods    = zeros(this.numdims,this.numhid);
                vishidinc  = zeros(this.numdims,this.numhid);
                hidbiasinc = zeros(1,this.numhid);
                visbiasinc = zeros(1,this.numdims);
                batchposhidprobs=zeros(this.numcases,this.numhid,this.numbatches);
            end            
            
            for epoch = epoch:this.maxepoch
                fprintf(1,'epoch %d\r',epoch);
                errsum=0;
                for batch = 1:this.numbatches
                    fprintf(1,'epoch %d batch %d\r',epoch,batch);
                    
                    visbias = repmat(visbiases,this.numcases,1);
                    hidbias = repmat(2*hidbiases,this.numcases,1);
                    %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    data = this.batchdata(:,:,batch);
                    data = data > rand(this.numcases,this.numdims);
                    
                    this.poshidprobs = 1./(1 + exp(-data*(2*vishid) - hidbias));
                    batchposhidprobs(:,:,batch)=this.poshidprobs;
                    this.posprods    = data' * this.poshidprobs;
                    poshidact   = sum(this.poshidprobs);
                    posvisact = sum(data);
                    
                    %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    poshidstates = this.poshidprobs > rand(this.numcases,this.numhid);
                    negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
                    negdata = negdata > rand(this.numcases,this.numdims);
                    this.neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - hidbias));
                    
                    this.negprods  = negdata'*this.neghidprobs;
                    neghidact = sum(this.neghidprobs);
                    negvisact = sum(negdata);
                    
                    %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    err= sum(sum( (data-negdata).^2 ));
                    errsum = err + errsum;
                    
                    if (epoch>5)
                        momentum=finalmomentum;
                    else
                        momentum=initialmomentum;
                    end
                    
                    %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    vishidinc = momentum*vishidinc + ...
                        epsilonw*( (this.posprods-this.negprods)/this.numcases - weightcost*vishid);
                    visbiasinc = momentum*visbiasinc + (epsilonvb/this.numcases)*(posvisact-negvisact);
                    hidbiasinc = momentum*hidbiasinc + (epsilonhb/this.numcases)*(poshidact-neghidact);
                    
                    vishid = vishid + vishidinc;
                    visbiases = visbiases + visbiasinc;
                    hidbiases = hidbiases + hidbiasinc;
                    %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if rem(batch,600)==0
                        figure(1);
                        this.dispims(negdata',28,28);
                        drawnow
                    end
                end
                fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
                
            end
            
            save fullmnistvh vishid visbiases hidbiases epoch

        end
        function this = rbm_l2(this)
            %% RBM_L2.m Training 2nd layer RBM with binary hidden and visible units
            % This program trains Restricted Boltzmann Machine in which
            % visible, binary, stochastic pixels are connected to
            % hidden, binary, stochastic feature detectors using symmetrically
            % weighted connections. Learning is done with 1-step Contrastive Divergence.
            % The program assumes that the following variables are set externally:
            % this.maxepoch  -- maximum number of epochs
            % this.numhid    -- number of hidden units
            % this.batchdata -- the data that is divided into batches (numcases numdims numbatches)
            % this.restart   -- set to 1 if learning starts from beginning
            
            if (this.restart ==1)
                
                epsilonw_0      = 0.05;   % Learning rate for weights
                epsilonvb_0     = 0.05;   % Learning rate for biases of visible units
                epsilonhb_0     = 0.05;   % Learning rate for biases of hidden units
                
                weightcost  = 0.001;
                initialmomentum  = 0.5;
                finalmomentum    = 0.9;
                
                load('fullmnistvh');
                vishid_l0 = vishid;
                hidbiases_l0 = hidbiases;
                visbiases_l0 = visbiases;
                [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
                numdims_l0 = this.numdims;
                
                this.numdims = this.numhid;
                this.numhid = this.numpen;
                
                this.restart=0;
                epoch=1;
                
                % Initializing symmetric weights and biases.
                vishid     = 0.01*randn(this.numdims, this.numhid);
                hidbiases  = zeros(1,this.numhid);
                visbiases  = zeros(1,this.numdims);
                
                this.poshidprobs = zeros(this.numcases,this.numhid);
                this.neghidprobs = zeros(this.numcases,this.numhid);
                this.posprods    = zeros(this.numdims,this.numhid);
                this.negprods    = zeros(this.numdims,this.numhid);
                vishidinc  = zeros(this.numdims,this.numhid);
                hidbiasinc = zeros(1,this.numhid);
                visbiasinc = zeros(1,this.numdims);
                
                numlab=10;
                labhid = 0.01*randn(numlab,this.numhid);
                labbiases  = zeros(1,numlab);
                labhidinc =  zeros(numlab,this.numhid);
                labbiasinc =  zeros(1,numlab);
                
                epoch=1;                
                
            end
            
            for epoch = epoch:this.maxepoch
                fprintf(1,'epoch %d\r',epoch);
                
                this.CD = ceil(epoch/20);
                
                epsilonw  = epsilonw_0/(1*this.CD);
                epsilonvb = epsilonvb_0/(1*this.CD);
                epsilonhb = epsilonhb_0/(1*this.CD);
                
                errsum=0;
                for batch = 1:this.numbatches
                    fprintf(1,'epoch %d batch %d\r',epoch,batch);
                    
                    %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    data_l0 = this.batchdata(:,:,batch);
                    poshidprobs_l0 = 1./(1 + exp(-data_l0*(2*vishid_l0) - repmat(2*hidbiases_l0,this.numcases,1)));
                    data = poshidprobs_l0 > rand(this.numcases,this.numdims);
                    targets = this.batchtargets(:,:,batch);
                    
                    bias_hid= repmat(hidbiases,this.numcases,1);
                    bias_vis = repmat(2*visbiases,this.numcases,1);
                    bias_lab = repmat(labbiases,this.numcases,1);
                    
                    this.poshidprobs = 1./(1 + exp(-data*(vishid) - targets*labhid - bias_hid));
                    this.posprods    = data' * this.poshidprobs;
                    posprodslabhid = targets'*this.poshidprobs;
                    
                    poshidact   = sum(this.poshidprobs);
                    posvisact = sum(data);
                    poslabact   = sum(targets);
                    
                    %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    poshidprobs_temp = this.poshidprobs;
                    
                    %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    for cditer=1:this.CD
                        poshidstates = poshidprobs_temp > rand(this.numcases,this.numhid);
                        
                        totin = poshidstates*labhid' + bias_lab;
                        neglabprobs = exp(totin);
                        neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlab));
                        
                        xx = cumsum(neglabprobs,2);
                        xx1 = rand(this.numcases,1);
                        neglabstates = neglabprobs*0;
                        for jj=1:this.numcases
                            index = find(xx1(jj) <= xx(jj,:), 1 ); % min(find(xx1(jj) <= xx(jj,:)));
                            neglabstates(jj,index) = 1;
                        end
                        xxx = sum(sum(neglabstates)) ;
                        
                        negdata = 1./(1 + exp(-poshidstates*(2*vishid)' - bias_vis));
                        negdata = negdata > rand(this.numcases,this.numdims);
                        poshidprobs_temp = 1./(1 + exp(-negdata*(vishid) - neglabstates*labhid - bias_hid));
                    end
                    this.neghidprobs = poshidprobs_temp;
                    
                    this.negprods  = negdata'*this.neghidprobs;
                    neghidact = sum(this.neghidprobs);
                    negvisact = sum(negdata);
                    neglabact = sum(neglabstates);
                    negprodslabhid = neglabstates'*this.neghidprobs;
                    
                    %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    err= sum(sum( (data-negdata).^2 ));
                    errsum = err + errsum;
                    
                    if (epoch>5)
                        momentum=finalmomentum;
                    else
                        momentum=initialmomentum;
                    end
                    
                    %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    vishidinc = momentum*vishidinc + ...
                        epsilonw*( (this.posprods-this.negprods)/this.numcases - weightcost*vishid);
                    labhidinc = momentum*labhidinc + ...
                        epsilonw*( (posprodslabhid-negprodslabhid)/this.numcases - weightcost*labhid);
                    
                    
                    visbiasinc = momentum*visbiasinc + (epsilonvb/this.numcases)*(posvisact-negvisact);
                    hidbiasinc = momentum*hidbiasinc + (epsilonhb/this.numcases)*(poshidact-neghidact);
                    labbiasinc = momentum*labbiasinc + (epsilonvb/this.numcases)*(poslabact-neglabact);
                    
                    
                    
                    vishid = vishid + vishidinc;
                    labhid = labhid + labhidinc;
                    
                    visbiases = visbiases + visbiasinc;
                    hidbiases = hidbiases + hidbiasinc;
                    labbiases = labbiases + labbiasinc;
                    
                end
                %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
                
                %%%% Look at the test scores %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if rem(epoch,10)==0
                    err = this.testerr(vishid_l0,hidbiases_l0,...
                        vishid,visbiases,hidbiases,labhid,labbiases);
                    fprintf(1,'Number of misclassified test examples: %d out of 10000 \n',err);
                end
                
                save fullmnistpo labhid labbiases vishid hidbiases visbiases epoch
                
            end
        end
        function this = dbm_mf(this)
            %% DBM_MF Joint training of all layers in a DBM 
            
            close all
            if (this.restart == 1)
                epsilonw      = 0.001;   % Learning rate for weights
                epsilonvb     = 0.001;   % Learning rate for biases of visible units
                epsilonhb     = 0.001;   % Learning rate for biases of hidden units
                weightcost    = 0.0002;
                initialmomentum  = 0.5;
                finalmomentum    = 0.9;
                
                [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
                
                numlab=10;
                this.numdim=this.numdims;
                
                this.restart=0;
                epoch=1;
                % Initializing symmetric weights and biases.
                
                vishid     = 0.001*randn(this.numdim, this.numhid);
                hidpen     = 0.001*randn(this.numhid,this.numpen);
                
                labpen = 0.001*randn(numlab,this.numpen);
                
                hidbiases  = zeros(1,this.numhid);
                visbiases  = zeros(1,this.numdim);
                penbiases  = zeros(1,this.numpen);
                labbiases  = zeros(1,numlab);
                
                this.poshidprobs = zeros(this.numcases,this.numhid);
                this.neghidprobs = zeros(this.numcases,this.numhid);
                this.posprods    = zeros(this.numdim,this.numhid);
                this.negprods    = zeros(this.numdim,this.numhid);
                
                
                vishidinc  = zeros(this.numdim,this.numhid);
                hidpeninc  = zeros(this.numhid,this.numpen);
                labpeninc =  zeros(numlab,this.numpen);
                
                
                hidbiasinc = zeros(1,this.numhid);
                visbiasinc = zeros(1,this.numdim);
                penbiasinc = zeros(1,this.numpen);
                labbiasinc = zeros(1,numlab);
                
                %%%% This code also adds sparcity penalty
                sparsetarget = .2;
                sparsetarget2 = .1;
                sparsecost = .001;
                sparsedamping = .9;
                
                hidbiases  = 0*log(sparsetarget/(1-sparsetarget))*ones(1,this.numhid);
                hidmeans = sparsetarget*ones(1,this.numhid);
                penbiases  = 0*log(sparsetarget2/(1-sparsetarget2))*ones(1,this.numpen);
                penmeans = sparsetarget2*ones(1,this.numpen);
                
                load('fullmnistpo.mat');
                
                hidpen = vishid;
                penbiases = hidbiases;
                visbiases_l2 = visbiases;
                labpen = labhid;
                clear labhid;
                
                load('fullmnistvh.mat');
                hidrecbiases = hidbiases;
                hidbiases = (hidbiases + visbiases_l2);
                epoch=1;
                
                this.neghidprobs = (rand(this.numcases,this.numhid));
                neglabstates = 1/10*(ones(this.numcases,numlab));
                data = round(rand(100,this.numdims));
                this.neghidprobs = 1./(1 + exp(-data*(2*vishid) - repmat(hidbiases,this.numcases,1)));
                
                epsilonw      = epsilonw/(1.000015^((epoch-1)*600));
                epsilonvb      = epsilonvb/(1.000015^((epoch-1)*600));
                epsilonhb      = epsilonhb/(1.000015^((epoch-1)*600));
                
                tot = 0;
            end
            
            for epoch = epoch:this.maxepoch
                [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
                
                fprintf(1,'epoch %d \t eps %f\r',epoch,epsilonw);
                errsum=0;
                
                [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
                
                counter=0;
                rr = randperm(this.numbatches);
                batch=0;
                for batch_rr = rr %1:this.numbatches,
                    batch=batch+1;
                    fprintf(1,'epoch %d batch %d\r',epoch,batch);
                    tot=tot+1;
                    epsilonw = max(epsilonw/1.000015,0.00010);
                    epsilonvb = max(epsilonvb/1.000015,0.00010);
                    epsilonhb = max(epsilonhb/1.000015,0.00010);
                    
                    
                    %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    data = this.batchdata(:,:,batch);
                    targets = this.batchtargets(:,:,batch);
                    data = double(data > rand(this.numcases,this.numdim));
                    
                    %%%%% First fo MF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    [this.poshidprobs, pospenprobs] = ...
                        this.mf(data,targets,vishid,hidbiases,visbiases,hidpen,penbiases,labpen,hidrecbiases);
                    
                    
                    bias_hid= repmat(hidbiases,this.numcases,1);
                    bias_pen = repmat(penbiases,this.numcases,1);
                    bias_vis = repmat(visbiases,this.numcases,1);
                    bias_lab = repmat(labbiases,this.numcases,1);
                    
                    this.posprods    = data' * this.poshidprobs;
                    posprodspen = this.poshidprobs'*pospenprobs;
                    posprodslabpen = targets'*pospenprobs;
                    
                    poshidact   = sum(this.poshidprobs);
                    pospenact   = sum(pospenprobs);
                    poslabact   = sum(targets);
                    posvisact = sum(data);
                    
                    
                    %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    negdata_CD1 = 1./(1 + exp(-this.poshidprobs*vishid' - bias_vis));
                    totin =  bias_lab + pospenprobs*labpen';
                    poslabprobs1 = exp(totin);
                    targetout = poslabprobs1./(sum(poslabprobs1,2)*ones(1,numlab));
                    [I,J]=max(targetout,[],2);
                    [I1,J1]=max(targets,[],2);
                    counter=counter+length(find(J==J1));
                    
                    
                    
                    %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    for iter=1:5
                        neghidstates = this.neghidprobs > rand(this.numcases,this.numhid);
                        
                        negpenprobs = 1./(1 + exp(-neghidstates*hidpen - neglabstates*labpen - bias_pen));
                        negpenstates = negpenprobs > rand(this.numcases,this.numpen);
                        
                        negdataprobs = 1./(1 + exp(-neghidstates*vishid' - bias_vis));
                        negdata = negdataprobs > rand(this.numcases,this.numdim);
                        
                        totin = negpenstates*labpen' + bias_lab;
                        neglabprobs = exp(totin);
                        neglabprobs = neglabprobs./(sum(neglabprobs,2)*ones(1,numlab));
                        
                        xx = cumsum(neglabprobs,2);
                        xx1 = rand(this.numcases,1);
                        neglabstates = neglabstates*0;
                        for jj=1:this.numcases
                            index = find(xx1(jj) <= xx(jj,:), 1); % min(find(xx1(jj) <= xx(jj,:)));
                            neglabstates(jj,index) = 1;
                        end
                        xxx = sum(sum(neglabstates)) ;
                        
                        totin = negdata*vishid + bias_hid + negpenstates*hidpen';
                        this.neghidprobs = 1./(1 + exp(-totin));
                        
                    end
                    negpenprobs = 1./(1 + exp(-this.neghidprobs*hidpen - neglabprobs*labpen - bias_pen));
                    
                    this.negprods  = negdata'*this.neghidprobs;
                    negprodspen = this.neghidprobs'*negpenprobs;
                    neghidact = sum(this.neghidprobs);
                    negpenact = sum(negpenprobs);
                    negvisact = sum(negdata);
                    neglabact = sum(neglabstates);
                    negprodslabpen = neglabstates'*negpenprobs;
                    
                    
                    %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    err= sum(sum( (data-negdata_CD1).^2 ));
                    errsum = err + errsum;
                    
                    if (epoch>5)
                        momentum=finalmomentum;
                    else
                        momentum=initialmomentum;
                    end
                    
                    %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    visbiasinc = momentum*visbiasinc + (epsilonvb/this.numcases)*(posvisact-negvisact);
                    labbiasinc = momentum*labbiasinc + (epsilonvb/this.numcases)*(poslabact-neglabact);
                    
                    hidmeans = sparsedamping*hidmeans + (1-sparsedamping)*poshidact/this.numcases;
                    sparsegrads = sparsecost*(repmat(hidmeans,this.numcases,1)-sparsetarget);
                    
                    penmeans = sparsedamping*penmeans + (1-sparsedamping)*pospenact/this.numcases;
                    sparsegrads2 = sparsecost*(repmat(penmeans,this.numcases,1)-sparsetarget2);
                    
                    labpeninc = momentum*labpeninc + ...
                        epsilonw*( (posprodslabpen-negprodslabpen)/this.numcases - weightcost*labpen);
                    
                    vishidinc = momentum*vishidinc + ...
                        epsilonw*( (this.posprods-this.negprods)/this.numcases - weightcost*vishid - ...
                        data'*sparsegrads/this.numcases );
                    hidbiasinc = momentum*hidbiasinc + epsilonhb/this.numcases*(poshidact-neghidact) ...
                        -epsilonhb/this.numcases*sum(sparsegrads);
                    
                    hidpeninc = momentum*hidpeninc + ...
                        epsilonw*( (posprodspen-negprodspen)/this.numcases - weightcost*hidpen - ...
                        this.poshidprobs'*sparsegrads2/this.numcases - (pospenprobs'*sparsegrads)'/this.numcases );
                    penbiasinc = momentum*penbiasinc + epsilonhb/this.numcases*(pospenact-negpenact) ...
                        -epsilonhb/this.numcases*sum(sparsegrads2);
                    
                    vishid = vishid + vishidinc;
                    hidpen = hidpen + hidpeninc;
                    labpen = labpen + labpeninc;
                    visbiases = visbiases + visbiasinc;
                    hidbiases = hidbiases + hidbiasinc;
                    penbiases = penbiases + penbiasinc;
                    labbiases = labbiases + labbiasinc;
                    %%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if (rem(batch,50)==0)
                        figure(1);
                        this.dispims(negdata',28,28);
                    end
                    
                end
                fprintf(1, 'epoch %4i reconstruction error %6.1f \n Number of misclassified training cases %d (out of 60000) \n', epoch, errsum,60000-counter);
                
                save  fullmnist_dbm labpen labbiases hidpen penbiases vishid hidbiases visbiases epoch;
                
            end
        end
        function [temp_h1,temp_h2,this] = ...
                mf(this, data,targets,vishid,hidbiases,visbiases,hidpen,penbiases,labpen,hidrecbiases)
            
            %% MF.m Implements mean-field inference
            
            [this.numdim, this.numhid]=size(vishid);
            [this.numhid, this.numpen]=size(hidpen);
            
            this.numcases = size(data,1);
            bias_hid = repmat(hidbiases,this.numcases,1);
            bias_pen = repmat(penbiases,this.numcases,1);
            big_bias = data*vishid;
            lab_bias = targets*labpen;
            
            temp_h1 = 1./(1 + exp(-data*(2*vishid) - repmat(hidbiases,this.numcases,1)));
            temp_h2 = 1./(1 + exp(-temp_h1*hidpen - targets*labpen - bias_pen));
            
            temp_h1_old = temp_h1;
            temp_h2_old = temp_h2;
            
            for ii= 1:this.meanFieldUpdates
                totin_h1 = big_bias + bias_hid + (temp_h2*hidpen');
                temp_h1_new = 1./(1 + exp(-totin_h1));
                
                totin_h2 =  (temp_h1_new*hidpen + bias_pen + lab_bias);
                temp_h2_new = 1./(1 + exp(-totin_h2));
                
                diff_h1 = sum(sum(abs(temp_h1_new - temp_h1),2))/(this.numcases*this.numhid);
                diff_h2 = sum(sum(abs(temp_h2_new - temp_h2),2))/(this.numcases*this.numpen);
                fprintf(1,'\t\t\t\tii=%d Mean-Field: h1=%f h2=%f\r',ii,diff_h1,diff_h2);
                if (diff_h1 < 0.0000001 & diff_h2 < 0.0000001)
                    break;
                end
                temp_h1 = temp_h1_new;
                temp_h2 = temp_h2_new;
            end
            
            temp_h1 = temp_h1_new;
            temp_h2 = temp_h2_new;            
        end
        function this = backprop(this)
            %% BACKPROP.m Backpropagation for fine-tuning a DBM             
            
            test_err=[];
            test_crerr=[];
            train_err=[];
            train_crerr=[];
            
            fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
            fprintf(1,'60 batches of 1000 cases each. \n');
            
            [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
            N=this.numcases;
            
            load('fullmnist_dbm'); %#ok<*LOAD>
            [this.numdims,this.numhids] = size(vishid);
            [this.numhids,this.numpens] = size(hidpen);
            
            %%%%%% Preprocess the data %%%%%%%%%%%%%%%%%%%%%%
            
            [testnumcases,testnumdims,testnumbatches]=size(this.testbatchdata);
            N=testnumcases;
            temp_h2_test = zeros(testnumcases,this.numpens,testnumbatches);
            for batch = 1:testnumbatches
                data = [this.testbatchdata(:,:,batch)];
                [temp_h1,temp_h2] = ...
                    this.mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases);
                temp_h2_test(:,:,batch) = temp_h2;
            end
            
            [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
            N=this.numcases;
            temp_h2_train = zeros(this.numcases,this.numpens,this.numbatches);
            for batch = 1:this.numbatches
                data = [this.batchdata(:,:,batch)];
                [temp_h1,temp_h2] = ...
                    this.mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases);
                temp_h2_train(:,:,batch) = temp_h2;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            w1_penhid = hidpen';
            w1_vishid = vishid;
            w2 = hidpen;
            h1_biases = hidbiases; h2_biases = penbiases;
            
            w_class = 0.1*randn(this.numpens,10);
            topbiases = 0.1*randn(1,10);
            
            for epoch = 1:this.maxepoch
                
                %%%% TEST STATS
                %%%% Error rates
                [testnumcases,testnumdims,testnumbatches]=size(this.testbatchdata);
                N=testnumcases;
                bias_hid= repmat(h1_biases,N,1);
                bias_pen = repmat(h2_biases,N,1);
                bias_top = repmat(topbiases,N,1);
                
                err=0;
                err_cr=0;
                counter=0;
                for batch = 1:testnumbatches
                    data = [this.testbatchdata(:,:,batch)];
                    temp_h2 = temp_h2_test(:,:,batch);
                    target = [this.testbatchtargets(:,:,batch)];
                    
                    w1probs = 1./(1 + exp(-data*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
                    w2probs = 1./(1 + exp(-w1probs*w2 - bias_pen));
                    targetout = exp(w2probs*w_class + bias_top );
                    targetout = targetout./repmat(sum(targetout,2),1,10);
                    [I,J]=max(targetout,[],2);
                    [I1,J1]=max(target,[],2);
                    counter=counter+length(find(J~=J1));
                    err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
                end
                
                test_err(epoch)=counter;
                test_crerr(epoch)=err_cr;
                fprintf(1,'\nepoch %d test  misclassification err %d (out of 10000),  test cross entropy error %f \n',epoch,test_err(epoch),test_crerr(epoch));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %%%% TRAINING STATS
                %%%% Error rates
                [this.numcases,this.numdims,this.numbatches]=size(this.batchdata);
                N=this.numcases;
                err=0;
                err_cr=0;
                counter=0;
                for batch = 1:this.numbatches
                    data = [this.batchdata(:,:,batch)];
                    temp_h2 = temp_h2_train(:,:,batch);
                    target = [this.batchtargets(:,:,batch)];
                    
                    w1probs = 1./(1 + exp(-data*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
                    w2probs = 1./(1 + exp(-w1probs*w2 - bias_pen));
                    targetout = exp(w2probs*w_class + bias_top );
                    targetout = targetout./repmat(sum(targetout,2),1,10);
                    [I,J]=max(targetout,[],2);
                    [I1,J1]=max(target,[],2);
                    counter=counter+length(find(J~=J1));
                    
                    err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
                end
                
                train_err(epoch)=counter;
                train_crerr(epoch)=err_cr;
                fprintf(1,'epoch %d train misclassification err %d train (out of 60000), train cross entropy error %f \n',epoch, train_err(epoch),train_crerr(epoch));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                save backprop_weights w1_vishid w1_penhid w2 w_class h1_biases h2_biases topbiases test_err test_crerr train_err train_crerr
                
                %%% Do Conjugate Gradient Optimization
                
                rr = randperm(600);
                for batch = 1:this.numbatches/100
                    fprintf(1,'epoch %d batch %d\r',epoch,batch);
                    data = zeros(10000,this.numdims);
                    temp_h2 = zeros(10000,this.numpens);
                    targets = zeros(10000,10);
                    tt1=(batch-1)*100+1:batch*100;
                    for tt=1:100
                        data( (tt-1)*100+1:tt*100,:) = this.batchdata(:,:,rr(tt1(tt)));
                        temp_h2( (tt-1)*100+1:tt*100,:) = temp_h2_train(:,:,rr(tt1(tt)));
                        targets( (tt-1)*100+1:tt*100,:) = this.batchtargets(:,:,rr(tt1(tt)));
                    end
                    
                    %%%%%%%% DO CG with 3 linesearches
                    
                    VV = [w1_vishid(:)' w1_penhid(:)' w2(:)' w_class(:)' h1_biases(:)' h2_biases(:)' topbiases(:)']';
                    Dim = [this.numdims; this.numhids; this.numpens; ];
                    
                    % checkgrad('CG_MNIST_INIT',VV,10^-5,Dim,data,targets);
                    max_iter=3;
                    if epoch<6
                        [X,fX,num_iter,ecg_XX] = minimize(this,VV,'this.CG_MNIST_INIT',max_iter,Dim,data,targets,temp_h2);
                    else
                        [X,fX,num_iter,ecg_XX] = minimize(this,VV,'this.CG_MNIST',max_iter,Dim,data,targets,temp_h2);
                    end
                    w1_vishid = reshape(X(1:this.numdims*this.numhids),this.numdims,this.numhids);
                    xxx = this.numdims*this.numhids;
                    w1_penhid = reshape(X(xxx+1:xxx+this.numpens*this.numhids),this.numpens,this.numhids);
                    xxx = xxx+this.numpens*this.numhids;
                    w2 = reshape(X(xxx+1:xxx+this.numhids*this.numpens),this.numhids,this.numpens);
                    xxx = xxx+this.numhids*this.numpens;
                    w_class = reshape(X(xxx+1:xxx+this.numpens*10),this.numpens,10);
                    xxx = xxx+this.numpens*10;
                    h1_biases = reshape(X(xxx+1:xxx+this.numhids),1,this.numhids);
                    xxx = xxx+this.numhids;
                    h2_biases = reshape(X(xxx+1:xxx+this.numpens),1,this.numpens);
                    xxx = xxx+this.numpens;
                    topbiases = reshape(X(xxx+1:xxx+10),1,10);
                    xxx = xxx+10;                    
                end                
            end
        end
        function [f,df] = CG_MNIST(     this,VV,Dim,XX,target,temp_h2)
            %% CG_MNIST.m Conjugate Gradient optimization for fine-tuning a DBM
            
            this.numdims = Dim(1);
            this.numhids = Dim(2);
            this.numpens = Dim(3);
            N = size(XX,1);
            
            X=VV;
            % Do decomversion.
            w1_vishid = reshape(X(1:this.numdims*this.numhids),this.numdims,this.numhids);
            xxx = this.numdims*this.numhids;
            w1_penhid = reshape(X(xxx+1:xxx+this.numpens*this.numhids),this.numpens,this.numhids);
            xxx = xxx+this.numpens*this.numhids;
            hidpen = reshape(X(xxx+1:xxx+this.numhids*this.numpens),this.numhids,this.numpens);
            xxx = xxx+this.numhids*this.numpens;
            w_class = reshape(X(xxx+1:xxx+this.numpens*10),this.numpens,10);
            xxx = xxx+this.numpens*10;
            hidbiases = reshape(X(xxx+1:xxx+this.numhids),1,this.numhids);
            xxx = xxx+this.numhids;
            penbiases = reshape(X(xxx+1:xxx+this.numpens),1,this.numpens);
            xxx = xxx+this.numpens;
            topbiases = reshape(X(xxx+1:xxx+10),1,10);
            xxx = xxx+10;
            
            bias_hid= repmat(hidbiases,N,1);
            bias_pen = repmat(penbiases,N,1);
            bias_top = repmat(topbiases,N,1);
            
            w1probs = 1./(1 + exp(-XX*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
            w2probs = 1./(1 + exp(-w1probs*hidpen - bias_pen));
            targetout = exp(w2probs*w_class + bias_top );
            targetout = targetout./repmat(sum(targetout,2),1,10);
            
            f = -sum(sum( target(:,1:end).*log(targetout)));
            
            IO = (targetout-target(:,1:end));
            Ix_class=IO;
            dw_class =  w2probs'*Ix_class;
            dtopbiases = sum(Ix_class);
            
            Ix2 = (Ix_class*w_class').*w2probs.*(1-w2probs);
            dw2_hidpen =  w1probs'*Ix2;
            dw2_biases = sum(Ix2);
            
            Ix1 = (Ix2*hidpen').*w1probs.*(1-w1probs);
            dw1_penhid =  temp_h2'*Ix1;
            
            dw1_vishid = XX'*Ix1;
            dw1_biases = sum(Ix1);
            
            df = [dw1_vishid(:)' dw1_penhid(:)' dw2_hidpen(:)' dw_class(:)' dw1_biases(:)' dw2_biases(:)' dtopbiases(:)']';
        end
        function [f,df] = CG_MNIST_INIT(this,VV,Dim,XX,target,temp_h2)
            %% CG_MNIST_INIT.m Conjugate Gradient optimization for fine-tuning a DBM (training top-level 
            %  parameters, while holding low-level parameters fixed). 
            
            this.numdims = Dim(1);
            this.numhids = Dim(2);
            this.numpens = Dim(3);
            N = size(XX,1);
            
            X=VV;
            % Do decomversion.
            w1_vishid = reshape(X(1:this.numdims*this.numhids),this.numdims,this.numhids);
            xxx = this.numdims*this.numhids;
            w1_penhid = reshape(X(xxx+1:xxx+this.numpens*this.numhids),this.numpens,this.numhids);
            xxx = xxx+this.numpens*this.numhids;
            hidpen = reshape(X(xxx+1:xxx+this.numhids*this.numpens),this.numhids,this.numpens);
            xxx = xxx+this.numhids*this.numpens;
            w_class = reshape(X(xxx+1:xxx+this.numpens*10),this.numpens,10);
            xxx = xxx+this.numpens*10;
            hidbiases = reshape(X(xxx+1:xxx+this.numhids),1,this.numhids);
            xxx = xxx+this.numhids;
            penbiases = reshape(X(xxx+1:xxx+this.numpens),1,this.numpens);
            xxx = xxx+this.numpens;
            topbiases = reshape(X(xxx+1:xxx+10),1,10);
            xxx = xxx+10;
            
            bias_hid= repmat(hidbiases,N,1);
            bias_pen = repmat(penbiases,N,1);
            bias_top = repmat(topbiases,N,1);
            
            w1probs = 1./(1 + exp(-XX*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
            w2probs = 1./(1 + exp(-w1probs*hidpen - bias_pen));
            targetout = exp(w2probs*w_class + bias_top );
            targetout = targetout./repmat(sum(targetout,2),1,10);
            
            f = -sum(sum( target(:,1:end).*log(targetout)));
            
            IO = (targetout-target(:,1:end));
            Ix_class=IO;
            dw_class =  w2probs'*Ix_class;
            dtopbiases = sum(Ix_class);
            
            Ix2 = (Ix_class*w_class').*w2probs.*(1-w2probs);
            dw2_hidpen =  w1probs'*Ix2;
            dw2_biases = sum(Ix2);
            
            Ix1 = (Ix2*hidpen').*w1probs.*(1-w1probs);
            dw1_penhid =  temp_h2'*Ix1;
            dw1_vishid = XX'*Ix1;
            dw1_biases = sum(Ix1);
            
            dhidpen = 0*dw2_hidpen;
            dw1_penhid = 0*dw1_penhid;
            dw1_vishid = 0*dw1_vishid;
            dw2_biases = 0*dw2_biases;
            dw1_biases = 0*dw1_biases;
            
            df = [dw1_vishid(:)' dw1_penhid(:)' dw2_hidpen(:)' dw_class(:)' dw1_biases(:)' dw2_biases(:)' dtopbiases(:)']';
        end
        function [X, fX, i, XX] = minimize(this, X, f, length, P1, P2, P3, P4, P5) %#ok<INUSD,INUSL>
            %% MINIMIZE.m Conjugate gradient code. 
            %
            % Written by Carl E. Rasmussen            
            %
            % Minimize a continuous differentialble multivariate function. Starting point
            % is given by "X" (D by 1), and the function named in the string "f", must
            % return a function value and a vector of partial derivatives. The Polack-
            % Ribiere flavour of conjugate gradients is used to compute search directions,
            % and a line search using quadratic and cubic polynomial approximations and the
            % Wolfe-Powell stopping criteria is used together with the slope ratio method
            % for guessing initial step sizes. Additionally a bunch of checks are made to
            % make sure that exploration is taking place and that extrapolation will not
            % be unboundedly large. The "length" gives the length of the run: if it is
            % positive, it gives the maximum number of line searches, if negative its
            % absolute gives the maximum allowed number of function evaluations. The
            % function returns when either its length is up, or if no further progress can
            % be made (ie, we are at a minimum, or so close that due to numerical problems,
            % we cannot get any closer). If the function terminates within a few
            % iterations, it could be an indication that the function value and derivatives
            % are not consistent (ie, there may be a bug in the implementation of your "f"
            % function). The function returns the found solution "X", a vector of function
            % values "fX" indicating the progress made and "i" the number of iterations
            % (line searches or function evaluations, depending on the sign of "length")
            % used.
            %
            % Usage: [X, fX, i] = this.minimize(X, f, length, P1, P2, P3, P4, P5)
            %
            % See also: checkgrad
            %
            % Copyright (C) 2001 by Carl Edward Rasmussen. Date 2001-07-18
            
            RHO = 0.01;                            % a bunch of constants for line searches
            SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
            INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
            EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
            MAX = 20;                         % max 20 function evaluations per line search
            RATIO = 100;                                      % maximum allowed slope ratio
            
            argstr = [f, '(X'];                      % compose string used to call function
            for i = 1:(nargin - 4)
                argstr = [argstr, ',P', int2str(i)]; %#ok<*AGROW>
            end
            argstr = [argstr, ')'];
            
            if (length>0)
                S=['Linesearch']; 
            else
                S=['Function evaluation']; 
            end %#ok<*NBRAK>
            
            i = 0;                                            % zero the run length counter
            ls_failed = 0;                             % no previous line search has failed
            fX = [];
            XX = [];
            try                
                [f1,df1] = eval(argstr);                      % get function value and gradient
            catch ME
                fprintf('mlsalakhudtinov.DBM.minimize:  [f1,df1] = eval(%s)\n', argstr);
                handerror(ME, 'mlsalakhudtinov:unexpectedParamValue', 'argstr->%s', argstr);
            end
            XX=vertcat(XX,[1,-f1]);  %by Russ, get first value
            i = i + (length<0);                                            % count epochs?!
            s = -df1;                                        % search direction is steepest
            d1 = -s'*s;                                                 % this is the slope
            z1 = 1/(1-d1);                                      % initial step is 1/(|s|+1)
            
            while i < abs(length)                                      % while not finished
                i = i + (length>0);                                      % count iterations?!
                
                X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
                X = X + z1*s;                                             % begin line search
                [f2,df2] = eval(argstr);
                i = i + (length<0);                                          % count epochs?!
                d2 = df2'*s;
                f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
                if length>0, M = MAX; else M = min(MAX, -length-i); end
                success = 0; limit = -1;                     % initialize quanteties
                while 1
                    while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) & (M > 0) %#ok<*AND2,*OR2>
                        limit = z1;                                         % tighten the bracket
                        if f2 > f1
                            z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
                        else
                            A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
                            B = 3*(f3-f2)-z3*(d3+2*d2);
                            z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
                        end
                        if isnan(z2) | isinf(z2)
                            z2 = z3/2;                  % if we had a numerical problem then bisect
                        end
                        z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
                        z1 = z1 + z2;                                           % update the step
                        X = X + z2*s;
                        [f2,df2] = eval(argstr);
                        M = M - 1; i = i + (length<0);                           % count epochs?!
                        d2 = df2'*s;
                        z3 = z3-z2;                    % z3 is now relative to the location of z2
                    end
                    if f2 > f1+z1*RHO*d1 | d2 > -SIG*d1
                        break;                                                % this is a failure
                    elseif d2 > SIG*d1
                        success = 1; break;                                             % success
                    elseif M == 0
                        break;                                                          % failure
                    end
                    A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
                    B = 3*(f3-f2)-z3*(d3+2*d2);
                    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
                    if ~isreal(z2) | isnan(z2) | isinf(z2) | z2 < 0   % num prob or wrong sign?
                        if limit < -0.5                               % if we have no upper limit
                            z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
                        else
                            z2 = (limit-z1)/2;                                   % otherwise bisect
                        end
                    elseif (limit > -0.5) & (z2+z1 > limit)          % extraplation beyond max?
                        z2 = (limit-z1)/2;                                               % bisect
                    elseif (limit < -0.5) & (z2+z1 > z1*EXT)       % extrapolation beyond limit
                        z2 = z1*(EXT-1.0);                           % set to extrapolation limit
                    elseif z2 < -z3*INT
                        z2 = -z3*INT;
                    elseif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT))   % too close to limit?
                        z2 = (limit-z1)*(1.0-INT);
                    end
                    f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
                    z1 = z1 + z2; X = X + z2*s;                      % update current estimates
                    [f2,df2] = eval(argstr);
                    M = M - 1; i = i + (length<0);                             % count epochs?!
                    d2 = df2'*s;
                end                                                      % end of line search
                
                if success                                         % if line search succeeded
                    f1 = f2; fX = [fX' f1]';
                    fprintf('%s %6i;  Value %4.6e\r', S, i, f1);
                    XX=vertcat(XX,[i,-f1]); %minus here for our purpuses only
                    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
                    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
                    d2 = df1'*s;
                    if d2 > 0                                      % new slope must be negative
                        s = -df1;                              % otherwise use steepest direction
                        d2 = -s'*s;
                    end
                    z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
                    d1 = d2;
                    ls_failed = 0;                              % this line search did not fail
                else
                    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
                    if ls_failed | i > abs(length)          % line search failed twice in a row
                        break;                             % or we ran out of time, so we give up
                    end
                    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
                    s = -df1;                                                    % try steepest
                    d1 = -s'*s;
                    z1 = 1/(1-d1);
                    ls_failed = 1;                                    % this line search failed
                end
            end
            fprintf('\r');
        end
        function err = testerr(this,vishid_l0,hidbiases_l0,vishid,visbiases,hidbiases,labhid,labbiases)
            %% TESTERR.m Computes misclassification error on the MNIST dataset. 
            
            [this.numdim,this.numhid]=size(vishid_l0);
            [this.numcases,this.numdims,this.numbatches]=size(this.testbatchdata);
            counter=zeros(10,10000);
            
            targets_all = zeros(10000,10);
            for batch=1:100
                targets_all( (batch-1)*100+1:batch*100,:) = this.testbatchtargets(:,:,batch);
            end
            
            bias_hid_l0= repmat(2*hidbiases_l0,this.numcases,1);
            bias_pen = repmat(hidbiases,this.numcases,1);
            
            for batch= 1:this.numbatches
                inter = zeros(this.numcases,10);
                data = this.testbatchdata(:,:,batch);
                
                totin_h1 = data*(2*vishid_l0) + bias_hid_l0;
                temp_h1 = 1./(1 + exp(-totin_h1));
                
                
                for tt=1:10
                    targets = zeros(this.numcases,10);
                    targets(:,tt)=1;
                    lab_bias =  targets*labhid;
                    
                    temp1 = temp_h1*visbiases' + targets*labbiases';
                    prod_3 = ones(this.numcases,1)*hidbiases + (temp_h1*vishid + targets*labhid);
                    p_vl  = temp1 + sum(log(1+exp(prod_3)),2);
                    inter(:,tt) = p_vl;
                end
                
                counter(:,(batch-1)*100+1:batch*100)=inter';
            end
            [I,J]=max(counter',[],2);
            [I1,J1]=max(targets_all,[],2);
            err1=length(find(J~=J1));
            % fprintf(1,'err %d\n',err1);
            err = err1;            
        end
        
 		function this = DBM(varargin)
 			%% DBM
 			%  Usage:  this = DBM()
            
        end
    end
    
    methods (Static)
        function [imdisp] = dispims(imstack,drows,dcols,flip,border,n2,fud)
            %% DISPIMS.m Displays progress during DBM training. 
            % [imdisp] = this.dispims(imstack,drows,dcols,flip,border,frame_rows,fud)
            %
            % display a stack of images
            % Originally written by Sam Roweis
            
            
            [pp,N] = size(imstack);
            if (nargin<8); fud=0; end
            if (nargin<7); n2=ceil(sqrt(N)); end
            
            if (nargin<4); dcols=drows; end
            if (nargin<5); flip=0; end
            if (nargin<6); border=2; end
            
            drb=drows+border;
            dcb=dcols+border;
            
            imdisp=min(imstack(:))+zeros(n2*drb,ceil(N/n2)*dcb);
            
            for nn=1:N
                
                ii=rem(nn,n2); if(ii==0); ii=n2; end
                jj=ceil(nn/n2);
                
                if(flip)
                    daimg = reshape(imstack(:,nn),dcols,drows)';
                else
                    daimg = reshape(imstack(:,nn),drows,dcols);
                end
                
                imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border))=daimg';
                
            end
            
            if(fud)
                imdisp=flipud(imdisp);
            end
            
            imagesc(imdisp); colormap gray; axis equal; axis off;
            drawnow;
        end
    end 
    
    %% PROTECTED
    
    methods (Access = protected)
        function  [temp_h1,temp_h2,this] = ...
                mf_class(this,data,vishid,hidbiases,visbiases,hidpen,penbiases)
            %% MF_CLASS.m Helper function used by backprop.m
            
            [this.numdim, this.numhid]=size(vishid);
            [this.numhid, this.numpen]=size(hidpen);
            
            this.numcases = size(data,1);
            bias_hid= repmat(hidbiases,this.numcases,1);
            bias_pen = repmat(penbiases,this.numcases,1);
            big_bias =  data*vishid;
            
            temp_h1 = 1./(1 + exp(-data*(2*vishid) - repmat(hidbiases,this.numcases,1)));
            temp_h2 = 1./(1 + exp(-temp_h1*hidpen - bias_pen));
            
            for ii= 1:50
                totin_h1 = big_bias + bias_hid + (temp_h2*hidpen');
                temp_h1_new = 1./(1 + exp(-totin_h1));
                
                totin_h2 =  (temp_h1_new*hidpen + bias_pen);
                temp_h2_new = 1./(1 + exp(-totin_h2));
                
                diff_h1 = sum(sum(abs(temp_h1_new - temp_h1),2))/(this.numcases*this.numhid);
                diff_h2 = sum(sum(abs(temp_h2_new - temp_h2),2))/(this.numcases*this.numpen);
%                fprintf(1,'\t\t\t\tii=%d h1=%f h2=%f\r',ii,diff_h1,diff_h2);
                if (diff_h1 < 0.0000001 & diff_h2 < 0.0000001)
                    break;
                end
                temp_h1 = temp_h1_new;
                temp_h2 = temp_h2_new;
            end
            
            temp_h1 = temp_h1_new;
            temp_h2 = temp_h2_new;
            
        end
    end

	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphy
 end

