# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    # fade: bool = False,
    mlkv: bool = False,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        # fade=fade,
        mlkv=mlkv,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """Nanashi Mumei (七詩ムメイ) is an English-language Virtual YouTuber associated with hololive. She debuted in 2021 as part of hololive -Council-, the second generation of members of hololive English, alongside Tsukumo Sana, Ceres Fauna, Ouro Kronii and Hakos Baelz.


        Contents
        1	Introduction Video
        2	Profile
        3	Personality
        4	Appearance
        5	History
        5.1	Background
        5.1.1	Debut
        5.2	2021
        5.3	2022
        5.4	2023
        6	Discography
        7	Events
        8	Mascots and Fans
        9	Relationships
        10	Quotes
        11	Trivia
        11.1	Name
        11.2	Lore
        11.3	Likes and dislikes
        11.4	Miscellaneous
        12	External links
        12.1	Media
        12.2	Twitter hashtags
        12.3	Statistics
        12.4	Further readings
        13	References
        ADVERTISEMENT
        Introduction Video
        【DEBUT_STREAM】Oh?_OH!_-holoCouncil_-hololiveEnglish
        【DEBUT STREAM】Oh? OH! -holoCouncil -hololiveEnglish

        Nanashi Mumei's introduction.

        Profile
        A member of the Council and the Guardian of "Civilization," a concept crafted by mankind.

        As a living embodiment of the sum of mankind's efforts—the mark that humans have left on the world—she is far removed from her fellow members, as well as other lifeforms. Due to not being created by the Gods, she was free to choose her own appearance, and decided to make herself owl-like, after the bird that symbolizes wisdom.

        She is gentle, wise, and an unbelievably hard worker. As a well-traveled vagabond, she is blessed with a wealth of knowledge of the world. She has seen, heard, and experienced so many things that she has forgotten most of them, one of them being her own name.

        For some reason, she seems to project a rather pitiable aura. Perhaps this is in part thanks to the loneliness she has often felt in her perennial travels. That is what gave her the idea of making her own friend out of a material that was indispensable to the development of human civilization: paper.

        "It may fade and rip, but once a friend, forever a friend."

        Personality
        At debut, Mumei gave the impression of a cheerful, gentle girl who is soft-spoken and somewhat innocent, especially when compared to her peers. She was initially very shy and often visibly nervous during streams.

        However, as she has gained confidence in streaming, a different persona has emerged. Mumei’s streams are now characterized by the duality between a relatively low energy, relaxing, and comfortable tone to another side which is more spontaneous and energetic; with her often demonstrating psychopathic tendencies in these moments. Though the disparity between her macabre thoughts and cute voice surprised many at first, the line separating these two sides has since blurred to a point of seamlessness – with Mumei’s appreciation of the grotesque and ‘cursed’ being wholly embraced by Hoomans today.

        This can be seen clearly whenever Mumei is asked to draw something; no matter what the original subject was supposed to be, Mumei will often end up drawing something Tim Burton-esque at best, and outright demonic at worst. Some fans theorize that this is simply what observing the accumulated sins of human civilization for thousands of years does to a person.[5]

        In stark contrast to her psychopathic tendencies, an early instance of Mumei being overwhelmed in a heartwarming manner was her monetization celebration stream, in which she chose not to read chat at the beginning because she knew that she would (and eventually did) get startled by the fast rate of donations sent to her. However, this same stream also highlighted the care and appreciation she holds for her fans, with her stating numerous times throughout the stream that she sees the precious time people set aside to watch her streams “genuinely as valuable [as money]”.[6] Similar sentiments were also shared during Mumei's membership opening stream in which she said that she didn't mind if Hoomans didn't become members as she was "so appreciative of... anybody who comes and watches", and also was understanding of viewers who wanted to wait until she was feeling better before purchasing membership because she was feeling notably unwell at the time of this stream - this choice making "a lot of sense" to her.[7]

        Outside of these celebration streams, her appreciation can further be seen in frequent Tweets expressing her gratefulness to those who watch and convey their enjoyment of her streams (particularly when the stream does not go as planned), her responses to those expressing gratitude for her content during streams, and her general interactions with chat. Such interactions include her pleading with tired fans staying up late for her stream to go get some sleep, as well as her expressing the happiness she felt knowing Hoomans were inspired to draw because of her.

        Although she has moments of high energy and excitability (especially while under the effects of caffeine), Mumei regards herself as a "low energy" individual with the social aspect of streaming being surprisingly tiring at times. She believes herself to be an awkward individual and this manifests itself at times where she struggles to come up with topics to talk about; with her often resorting to impromptu singing or noises to fill the silence.[8] These bouts of random sounds and responses may also be partially attributed to her birdbrained nature, with Mumei herself admitting that she has a very short attention span and finds it difficult to multitask.

        At debut, Hakos Baelz described Mumei as having the cutest voice in hololive -Council-, though she has also demonstrated a rather unexpectedly wide vocal range, along with a tendency to emit high-pitched screeches when surprised or agitated.[9] Her fans and even other members of hololive, particularly Ceres Fauna, have developed some degree of protectiveness for her, due to her soft nature giving a "little sister" vibe, though Mumei's later behavior would eventually create uncertainty as to who needs to be protected from whom.

        Appearance
        Nanashi Mumei has golden-brown eyes and long, brown ponytail hair with a pair of feathers atop her head. She dresses in a brown cloak secured with a ribbon. Mumei also wears gloves that are fingerless on her index and pinky fingers. She has a belt wrapped around her waist with a lantern, pouch, and small dagger around it. Below her outfit is a pleated skirt. She has leggings with her right leg reaching above her thighs and the left leg below her thighs, and wears brown boots. Friend flying beside her is a paper bag with a drawn mouth and a plaster shaped like a cross on his face.

        History
        Background
        Following the success of hololive English in 2020 and 2021, COVER Corporation announced auditions for a second batch of English members from 12 February to 26 March 2021.[10] Twitter accounts for the five new members were created in June 2021 and YouTube accounts created on 26 July 2021.

        Teaser videos for the new generation began on the official hololive English YouTube Channel in August 2021. On 1 August, "Prelude" revealed a story of gods creating the four concepts of space, nature, time, and civilization.[11] On 14 August, "Omen" described the creation of avatars of those four concepts plus that of the primordial force of Chaos, revealing five silhouettes.[12] On 17 August, "Council" revealed the five new hololive English members and announced a grand debut scheduled for the coming weekend.[13] Although the traditional term "generation" was not officially used to refer to the new members, who had been referred to as "hololive English Generation 2" by fans, the new members nonetheless formed a group known as "-Council-".

        Upon conclusion of the Council debut PV, all five new members' Twitter and YouTube accounts were revealed, and each made their first tweet.[14][15][16][17][18] Twitter quickly restricted Nanashi Mumei and Ouro Kronii's accounts due to the sudden increase in followers.[19]

        On 17 August 2021, around 24 hours after the initial announcement, Nanashi Mumei reached 100,000 YouTube subscribers before her debut. However, YouTube soon removed around half of this number, possibly due to the sudden subscriber growth on a channel with no videos yet, which may have triggered a YouTube anti-bot algorithm.

        Debut
        Nanashi Mumei's debut was scheduled for 22 August. However, the -Council- debuts were postponed for 24 hours due to unspecified technical issues,[20] which members jokingly blamed on the "EN curse", a perceived tendency for hololive English members to suffer from unexpected technical problems.[21] The debuts were rescheduled at the same time on Sunday 22 August (US/Europe time) or Monday 23 August (Japan time).[22]

        Mumei made her debut stream on Monday 23 August at 6:30 AM JST (22 August 2021 at 2:30 PM PDT, 10:30 PM BST). She was the fourth member of -Council- to debut. The stream had a peak viewer count of over 90,000, and she reached over 100,000 subscribers by the start of the stream. She previewed an unnamed original song.[23]

        2021
        On 26 August, Mumei reached 200,000 YouTube subscribers. She had previously reached this milestone on 23 August, but a number of subscribers had been removed by a YouTube anti-bot algorithm.

        On 5 September, Mumei's channel got approved for monetization.

        On 22 September, Mumei reached 300,000 YouTube subscribers.

        On 24 September, her channel became open for memberships.[24]

        On 27 September, Mumei held her first official collaboration with another hololive member outside her generation, playing Minecraft with Takanashi Kiara.[25]

        On 30 October, Mumei reached 400,000 YouTube subscribers.

        On 20 December, Mumei and the rest of hololive EN collaborated on a cover song "The Twelve Days of Christmas" by Frederic Austin, which was also her first official cover song.

        2022
        On 1 January, Mumei reached 500,000 YouTube subscribers. She is the 47th hololive member to reach this milestone.

        On 6 January, hololive's Twitter account announced that both Myth and Council members would get new year costumes.[26]

        On 15 January, Mumei debuted"""
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
