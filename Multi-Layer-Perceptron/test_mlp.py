import mlp as mlp

mlp1 = mlp.MLP([1,5,5,1],[2,2,3],0.1,2000,50,0.000000001,0.0000000001,4)
mlp1.set_training_set("patterns.txt","targets.txt",80)
mlp1.train()
mlp1.save_network('nn.pkl')
mlp2 = mlp.MLP.load_network('nn.pkl')
mlp2.feed_forward_propagate(4.2)
