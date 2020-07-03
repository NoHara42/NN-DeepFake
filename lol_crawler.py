# -*- coding: utf-8 -*-
import scrapy
import os
import urllib.request

path = "lol_data"

"""
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
"""

class LolSpider(scrapy.Spider):
    name = 'lol'
    allowed_domains = ['https://leagueoflegends.fandom.com/']
    start_urls = ['https://leagueoflegends.fandom.com/wiki/List_of_champions']

    def parse(self, response):
        wordList = ['Wiki-wordmark', 'data:image', 'slot1-images', 'Marksman', 'Fighter',
                    'Tank', 'Mage', 'Slayer', 'Controller', 'Specialist', 'BE_icon', 'RP_icon'
                    'Rainbow-Club-57-cover', 'Ao_ShinSquare', 'AvashaSquare', 'AverdrianSquare',
                    'CeeCeeSquare', 'Cyborg_CowboySquare', 'Eagle_RiderSquare', 'GavidSquare',
                    'Iron_EngineerSquare', 'IvanSquare', 'OmenSquare', 'PriscillaSquare', 'Rob_BlackbladeSquare',
                    'SethSquare', 'TabuSquare', 'TikiSquare', 'WellSquare', 'UrfSquare', 'N2losLW-asset-kids-mezzanine1-16x9-N70Tdc3',
                    'Spotlight-pic', 'RP_icon.png', 'club57']

        for item in response.css("a img::attr(src)").getall():
            if any(map(item.__contains__, wordList)):
                pass
            else:
                champion_name = item.split('revision',1)[0].split("/" ,8)[7].split("_",1)[0]
                file_name = "lol_data//" + champion_name + ".png"
                urllib.request.urlretrieve(item, file_name)
                #print(champion_name)
        