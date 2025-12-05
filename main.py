import data.torchvision as vision
import data.prepare_images as prep
import models.plant_model as plantModel
import data.loadSingleImage as loadImg
import training.train as train
import training.train as evaluate

RESET = "\033[0m"
Blue = "\033[34m"
Green = "\033[32m"
Red = "\033[31m"

def main():
    print(f"{Blue}========================================{RESET}")
    print("            MAIN MENU - CLI                           ")
    print(f"{Blue}========================================{RESET}")
    print(f"{Green}1.){RESET} Load Image to Analyze by AI        ")
    print(f"{Green}2.){RESET} Train Plantmodel                   ")
    print(f"{Green}3.){RESET} Validate Plantmodel                ")
    print(f"{Red}0.){RESET} Exit                                 ")
    print(f"{Blue}========================================{RESET}")
    choice = input()    
    if choice == "1":
        loadImg.loadSingleImage()
    elif choice == "2":
        train.train()
    elif choice == "3":
        evaluate.evaluate(model, device,dataloader, criterion)
    elif choice == "0":
        print("Exiting...")
        exit()
        return
main()
