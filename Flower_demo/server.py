"""
@Time : 2023/11/28 11:34
@Author : yanzx
@Description : 
"""
import sys, os
sys.path.append(os.getcwd())
import flwr as fl
from model import Net



def main():
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))


if __name__ == '__main__':
    main()
