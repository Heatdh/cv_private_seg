<script lang="ts">
  import './assets/normalize.css';

  import { Card } from 'flowbite-svelte';
  import { FingerPrint } from 'svelte-heros-v2';

  import FileUploader from './lib/FileUploader.svelte';
  import ResultDisplay from './lib/ResultDisplay.svelte';
  import type { Result } from './types';

  let done = false;
  let result: Result = {};
</script>

<main>
  <header>
    <FingerPrint size="48" color="#ffa6c1" />
    <h1>Might You Have Breast Cancer?</h1>
  </header>

  <div class="description">
    <p>{import.meta.env.VITE_DESCRIPTION_TEXT}</p>
  </div>

  <div class="center">
    <Card class="w-full">
      {#if !done}
        <FileUploader bind:done={done} bind:result={result} />
      {/if}
      <ResultDisplay result={result} />
    </Card>
  </div>
</main>

<style>
  main {
    height: 90vh;
    width: 60vw;
  }

  header {
    display: flex;
    justify-content: center;

    margin-bottom: 1rem;
  }

  .description {
    text-align: justify;

    margin-bottom: 1rem;
  }

  .center {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;

    height: 60%;
    width: 100%;
  }
</style>
