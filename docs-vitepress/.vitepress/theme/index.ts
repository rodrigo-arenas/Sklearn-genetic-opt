import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import HomeExtension from './components/HomeExtension.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-features-after': () => h(HomeExtension),
    })
  },
}
