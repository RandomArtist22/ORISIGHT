/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Fraunces', 'serif'],
        body: ['Source Sans 3', 'sans-serif']
      },
      colors: {
        ink: '#071a1f',
        slate: '#0f2f38',
        mint: '#6fd3c7',
        amber: '#f4b860',
        coral: '#e26b5b',
        fog: '#d9ece8'
      },
      boxShadow: {
        glass: '0 30px 80px -35px rgba(0, 0, 0, 0.7)'
      },
      keyframes: {
        rise: {
          '0%': { opacity: '0', transform: 'translateY(18px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        }
      },
      animation: {
        rise: 'rise 650ms cubic-bezier(0.2, 0.9, 0.2, 1) both'
      }
    }
  },
  plugins: []
}
