*******************************
*   SCC Expressive Strings    *
*   by S. Christian Collins   *
*        3/7/2016 v1.0        *
*******************************

* About *
SCC Expressive Strings is a virtual string ensemble programmed for maximum expressiveness. To get the most out of this library, it is recommended to have an expression pedal connected to an 88-key MIDI keyboard.

The samples have been collected from various free SoundFont banks I have found online. It is presumed that the samples contained within are royalty-free, but I can only go off of the information presented by the original SoundFont authors, which is scant at best. Here are the SoundFont banks from which SCC Expressive Strings draws its samples:

  . "DXS Super Orchestral Strings.sf2" by Xiaosu Du
  . "KBH Strings.sf2" by unknown author
  . "DXS Super Pizz.sf2" by Xiaosu Du
  . "Florestan Martellato Strings.sf2" by Nando Florestan
  . "Pizzicato Strings.sf2" by unknown author
  . I can no longer find the bank from which I got the tremolo samples.


* Instructions *
To play the instrument, you will need to load one of the two .sfz files into an SFZ-compatible sampler. I highly recommend the free Plogue sforzando plugin (https://www.plogue.com/products/sforzando/) for this task. I cannot guarantee that SCC Expressive Strings will work correctly with other SFZ-compatible samplers.

The filename suffix "CC1" or "CC11" determines which continuous controller will control the expression. CC1 is the mod wheel, and CC11 is the expression pedal. If you would like to use a different CC to control expression, make a copy of the "SCC Expressive Strings-Key Switch-CC11.sfz" file, open the copy in a text editor (not Windows Notepad... use Notepad++ instead) and search for all instances of "cc11" replacing them with the desired CC number (e.g. "cc2").

SCC Expressive Strings contains seven presets that are accessed via key switch. This means you must press the appropriate key on your MIDI keyboard to switch to the desired preset. This is great for quickly switching between different articulations like legato and staccato. Here are the keys and their associated presets (C4 = middle-C):

  A0: Legato (CC1 or CC11 controls expression)
    This is the default preset. Note velocity controls the quickness of the bowstroke, and the chosen expression method (CC1 or CC11) controls the volume and tone.

  A#0: Legato (velocity controls expression)
    This preset is for those without expression input. With this preset, the note velocity controls volume, tone, and the quickness of the bowstroke.

  B0: Staccato (velocity controls expression)

  A7: Tremolo (CC1 or CC11 controls expression)

  A#7: Tremolo (velocity controls expression)

  B7: Legato/Tremolo crossfade (CC1 or CC11 controls expression, soft pedal controls sample crossfade)
    This preset lets you switch between legato and tremolo samples using the soft pedal (CC67) to crossfade smoothly between the two. Requires a soft pedal that sends values gradually between 0-127, rather than a simple switch, which is either on (127) or off (0).

  C0: Pizzicato (velocity controls expression)


* License *
Due to the lack of any license information in the original samples, I am releasing this instrument as "use at your own risk". I have put a lot of work into programming this virtual instrument, so if you make a derivative work, a mention of my name and a link to my website (www.schristiancollins.com) would be appreciated.


* Changelog *
1.0
  . First release.