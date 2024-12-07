import matplotlib.pyplot as plt
import numpy as np


# Given a nested list of train/test loss/accuracy and training time and a list of 
# traversed value (with description), create and save a comparison plot 
def plot_comparison(trial_results, list_x, desc, save_path):
    
    avg_results = np.mean(trial_results, axis=0)
    std_devs = np.sqrt(np.var(trial_results, axis=0))
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list_x, avg_results[:,0], label='Train', color='blue')
    plt.plot(list_x, avg_results[:,2], label='Test', color='orange')
    
    # Plot the variance as a shaded region
    plt.fill_between(list_x , 
                 avg_results[:,0] - std_devs[:,0], 
                 avg_results[:,0] + std_devs[:,0], 
                 color='blue', alpha=0.2)
    
    plt.fill_between(list_x , 
                 avg_results[:,2] - std_devs[:,2], 
                 avg_results[:,2] + std_devs[:,2], 
                 color='orange', alpha=0.2)
    plt.xticks(list_x)
    plt.xlabel(desc)
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train/test loss comparison")
    fig.savefig(save_path+'loss.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list_x, avg_results[:,1], label='Train', color='blue')
    plt.plot(list_x, avg_results[:,3], label='Test', color='orange')
    
    # Plot the variance as a shaded region
    plt.fill_between(list_x , 
                 avg_results[:,1] - std_devs[:,1], 
                 avg_results[:,1] + std_devs[:,1], 
                 color='blue', alpha=0.2)
    
    plt.fill_between(list_x , 
                 avg_results[:,3] - std_devs[:,3], 
                 avg_results[:,3] + std_devs[:,3], 
                 color='orange', alpha=0.2)
    plt.xticks(list_x)
    plt.xlabel(desc)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train/test accuracy comparison")
    fig.savefig(save_path+'accuracy.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(list_x, avg_results[:,4], label='Training time', color='blue')
    plt.plot(list_x, avg_results[:,5], label='Test inference', color='orange')
    
    # Plot the variance as a shaded region
    plt.fill_between(list_x , 
                 avg_results[:,4] - std_devs[:,4], 
                 avg_results[:,4] + std_devs[:,4], 
                 color='blue', alpha=0.2)
    
    plt.fill_between(list_x , 
                 avg_results[:,5] - std_devs[:,5], 
                 avg_results[:,5] + std_devs[:,5], 
                 color='orange', alpha=0.2)
    plt.xticks(list_x)
    plt.xlabel(desc)
    plt.ylabel("Time in seconds")
    plt.legend()
    plt.title("Training and test inference time")
    fig.savefig(save_path+'time.png', bbox_inches='tight')
    plt.show()
    plt.close()
    



    
