export type EnvironmentTheme = {
  id: string;
  title: string;
  backdropAsset: string;
  gradientStart: string;
  gradientEnd: string;
  floorColor: string;
  accentColor: string;
  textColor: string;
};

const themes: EnvironmentTheme[] = [
  {
    id: 'neon-grid',
    title: 'Neon Grid',
    backdropAsset: 'assets/env/neon-grid.svg',
    gradientStart: '#050816',
    gradientEnd: '#0e1b38',
    floorColor: '#202644',
    accentColor: '#3cf0ff',
    textColor: '#d8f8ff',
  },
  {
    id: 'breach-factory',
    title: 'Breach Factory',
    backdropAsset: 'assets/env/breach-factory.svg',
    gradientStart: '#150c07',
    gradientEnd: '#3c180e',
    floorColor: '#3b2d21',
    accentColor: '#ff9f4a',
    textColor: '#ffe1bf',
  },
  {
    id: 'citadel-core',
    title: 'Citadel Core',
    backdropAsset: 'assets/env/citadel-core.svg',
    gradientStart: '#04130c',
    gradientEnd: '#0f3325',
    floorColor: '#193d2f',
    accentColor: '#62ffb8',
    textColor: '#d7ffe9',
  },
  {
    id: 'arena-default',
    title: 'Arena Prime',
    backdropAsset: 'assets/env/arena-default.svg',
    gradientStart: '#140d1f',
    gradientEnd: '#251534',
    floorColor: '#3c2b52',
    accentColor: '#f8dd62',
    textColor: '#fef6cb',
  },
];

const keywordMap: Array<{pattern: RegExp; themeId: string}> = [
  {pattern: /neon|grid|cyber|network/i, themeId: 'neon-grid'},
  {pattern: /factory|plant|industrial|forge/i, themeId: 'breach-factory'},
  {pattern: /citadel|fort|core|vault|secure/i, themeId: 'citadel-core'},
];

const hash = (value: string): number => {
  let result = 0;
  for (let i = 0; i < value.length; i += 1) {
    result = (result << 5) - result + value.charCodeAt(i);
    result |= 0;
  }

  return Math.abs(result);
};

export const getEnvironmentTheme = (env: string): EnvironmentTheme => {
  const matched = keywordMap.find((entry) => entry.pattern.test(env));
  if (matched) {
    const byId = themes.find((theme) => theme.id === matched.themeId);
    if (byId) {
      return byId;
    }
  }

  return themes[hash(env) % themes.length] ?? themes[0];
};
