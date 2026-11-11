import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import 'virtual:mathjax-styles.css'
import './style.css'
import './docstrings.css'
import './overrides.css'
// __DV_PLUGIN_COMPONENT_IMPORTS__

// DocumenterVitepress writes interactive HTML through v-html. Activate scripts
// after insertion, waiting for dependencies before the plot's inline script.
async function activateScripts(container: Element): Promise<void> {
  const figure = container.querySelector<HTMLElement>('[data-figure-src]')
  if (figure) {
    const response = await fetch(figure.dataset.figureSrc!)
    if (!response.ok) throw new Error(`Figure request failed: ${response.status}`)
    figure.innerHTML = await response.text()
  }
  for (const old of Array.from(container.querySelectorAll('script'))) {
    const fresh = document.createElement('script')
    for (const attr of Array.from(old.attributes)) fresh.setAttribute(attr.name, attr.value)
    fresh.textContent = old.textContent
    const ran = old.hasAttribute('src')
      ? new Promise<void>((resolve, reject) => {
          fresh.addEventListener('load', () => resolve(), { once: true })
          fresh.addEventListener('error', () => reject(new Error(`Could not load ${fresh.src}`)), { once: true })
        })
      : Promise.resolve()
    old.replaceWith(fresh)
    await ran
  }
}

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.directive('exec-scripts', {
      mounted(el: HTMLElement) { void activateScripts(el) },
    })
    // __DV_PLUGIN_COMPONENT_REGISTRATIONS__
  },
} satisfies Theme
