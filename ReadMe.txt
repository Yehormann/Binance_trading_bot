##Trading bot.
First step to run trading bot is to upload corresponding libraries via terminal.
1.pip install pandas
2.pip install datetime
4.pip install numpy
5.pip install matplotlib
6.pip install python-binance

The second step is to open the config.json file and insert your API keys. Also, configure the trade pairs in the 'symbol' line, with the initial symbol value set as 'BTCUSDT'.

To launch the trading bot, open the terminal and navigate to the root folder of the trading bot project. Then, type 'tradeBot.py' to start.

In the menu folder, choose between Backtesting and Online mode. For the Online mode, input your API key and secret key to configure the trading bot with your Binance account. Ensure that you provide comprehensive rights to the trading bot in your Binance account.

Troubleshooting:
If there is an error with the libraries, check your environment path.
If there is a problem with the API connection, verify the correct permissions in your Binance account.


##Disclaimer

The information provided in this code and any accompanying materials is for educational 
purposes only and should not be considered as financial or investment advice. 
Trading cryptocurrencies or any other financial instruments involves a significant level of risk, 
and it is possible to lose money. Before making any trading decisions, it is important to conduct
 your own research, assess your risk tolerance, and consult with a qualified financial advisor.
 The use of this code does not guarantee any specific trading outcomes, and the author(s) shall 
not be held responsible for any losses or damages incurred as a result of using this code. 
Trading should only be undertaken by individuals who understand the risks involved and are willing to 
accept full responsibility for their actions.

To stop the trading bot press Ctrl + C in terminal.
