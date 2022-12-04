const config = {
  content: [
    "./src/**/*.{html,js,svelte,ts}",
    "./node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}",
  ],

  theme: {
    extend: {
      maxWidth: {
        sm: '100%'
      }
    },
  },

  plugins: [
    require("flowbite/plugin"),
  ],
  darkMode: "class",
};

module.exports = config;
