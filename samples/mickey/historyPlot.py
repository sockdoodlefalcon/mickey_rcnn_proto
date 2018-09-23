import matplotlib.pyplot as plt
# Comparison plots for training and validation loss and accuracy for History array of the provided unit sizes
def plotHistoryArray(historyArray):

    clrs = ['g', 'b', 'brown', 'tomato', 'lawngreen', 'olive', 'gold', 'teal', 'r', 'm', 'c', 'k']
   
    plt.figure(1,figsize=(8, 15))
    plt.subplot(311)
    
    clrsidx = 0 
    for i in range(0, len(historyArray)):
        
        loss = historyArray[i].history['loss']
        val_loss = historyArray[i].history['val_loss']   
        epochs = range(1, len(loss) + 1)
        
        plt.plot(epochs, loss, clrs[clrsidx], label='Training loss ', linewidth=1.8)
        plt.plot(epochs, val_loss, clrs[clrsidx+1], label='Validation loss ', linewidth=1.8)
        clrsidx = clrsidx + 2
    
    
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  
    plt.grid(True)
    
    plt.subplot(312)
    
    clrsidx = 0 
    for i in range(0, len(historyArray)):
        acc = historyArray[i].history['acc']
        val_acc = historyArray[i].history['val_acc']
        epochs = range(1, len(acc) + 1)
        
        plt.plot(epochs, acc, clrs[clrsidx], label='Training accuracy ', linewidth=1.8)
        plt.plot(epochs, val_acc, clrs[clrsidx+1], label='Validation accuracy ', linewidth=1.8)
        clrsidx = clrsidx + 2
    
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.show()    
