import mlp as mlp

mlp1 = mlp.MLP([1,10,6,1],[2,1,3],0.2,1000,200,0.0000000001,0.0000000001,4)
mlp1.set_training_set("patterns.txt","targets.txt",80)
mlp1.train()