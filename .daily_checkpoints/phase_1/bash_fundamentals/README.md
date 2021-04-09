# Bash Fundamentals

Hello! Welcome to a quick challenge regarding bash fundamentals:

## Questions

1. **Run `sh lyrics.sh` from the project directory (i.e. `bash-challenge-seattle-ds-012720`)***
     + At this point, you should get the following error: 
     ```bash
     sh: lyrics.sh: No such file or directory
     ```
     + Your goal is the following:
         + Locate `lyrics.sh` within the project directory
         + Copy `lyrics.sh` from wherever it is, and paste it into the project directory
         + Re-run `sh lyrics.sh`
         + _Hint: run the command `tree` from your project directory and visually inspect the structure of the directory_
2. **Create a notebook called `steps.ipynb`, and, for each bash command that you run below, add a copy of that command to `steps.ipynb`.**
3. **Locate `not_lyrics.sh`, copy it into the project directory, and rename it to `other_lyrics.sh`**
4. **Create a file called `favorite_song.txt` that contains one sentence that declares your favorite song from The Beatles.**
    + _Hint: you'll need to use some combination of the command `touch` and the text editor `vim`_
5. **The file `the_beatles.txt` should contain four lines, one line per member of The Beatles. Currently, there are only two members listed. From the Terminal, append the remaining names of the two members to the file.**
    + _Hint: you'll need to store each member's name in a string before appending it to the file_
6. **Using `vim`, search and replace all occurrences of the text "golden watercraft" with "yellow submarine" in the file `yellow_submarine.txt`.** Hint [here](https://vim.fandom.com/wiki/Search_and_replace).
