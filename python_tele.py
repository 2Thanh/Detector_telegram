import asyncio
import telegram

def send_telegram():
    try:
        my_token = '5972711467:AAElDiSKR3ei2oUOjBTWlHOgPVvP-wOK9zw'
        bot = telegram.Bot(token = my_token)

        async def func():
            #await bot.send_message(chat_id='5559085957', text='g')
            #Change the path file detector image
            await bot.send_photo(chat_id='5559085957', photo= open('/home/thanh/Documents/Pandas/Detection_with_camera/alert.png','rb'))

        async def main():
            await func()
        asyncio.run(main())
    except:
        print("Cannot send message")
