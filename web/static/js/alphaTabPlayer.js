// loading the links to elements from the html file
const wrapper = document.getElementById('alphaTab');
const playPauseBtn = document.getElementById('playPause');

// setting up the alphaTab
const settings = {
  core: {
    // loading the font of the notes
    fontDirectory: 'https://cdn.jsdelivr.net/npm/@coderline/alphatab@latest/dist/font/'
  },
  display: {
    layoutMode: alphaTab.LayoutMode.Page,
    // show only the sheet music and not guitar tab
    staveProfile: alphaTab.StaveProfile.Score
  },
  player: {
    enablePlayer: true,
    // sound of the instruments
    soundFont: 'https://cdn.jsdelivr.net/npm/@coderline/alphatab@latest/dist/soundfont/sonivox.sf2'
  }
};

// turn on the alphaTab player
const api = new alphaTab.AlphaTabApi(wrapper, settings);

// get the URL adress from the HTML attribute and load it
//
// using .dataset finds all the attributes starting with data and
// also converts the name to camelCase
const midiUrl =  wrapper.dataset.midiUrl;

if (midiUrl){
    api.load(midiUrl);
}

// unlock the button when the soundFont is loaded
api.soundFontLoaded.on(() => {
  playPauseBtn.disabled = false;
});

// add event listener to play button
playPauseBtn.addEventListener('click', () => {
  api.playPause();
});