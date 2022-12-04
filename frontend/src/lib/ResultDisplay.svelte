<script lang="ts">
    import { Classification, type Result } from '../types';
    import Calendar from './Calendar.svelte';

    export let result: Result;
</script>

<div>
    {#if !!result.classification}
        <p>Your tumor has been classified as: <span class={result.classification}>{result.classification}</span></p>
    {/if}
    {#if result.classification === Classification.BENIGN}
        <p>You're likely all good! No need to worry for now. Regardless, we have provided a calendar with openings for you.</p>
    {:else if result.classification === Classification.MALIGNANT}
        <p>You should absolutely go see a doctor! Here's a calendar of openings for doctors in your area:</p>
    {:else if result.classification === Classification.MALIGNANT_WITH_CALLBACK}
        <p>You should probably go see a doctor... Here's a calendar of openings for doctors in your area:</p>
    {/if}
    <div class="calendar-container">
        <Calendar severity={result.classification || Classification.BENIGN} />
    </div>
</div>

<style>
    span {
        font-weight: bold;
    }
    span.BENIGN {
        color: green;
    }
    span.MALIGNANT {
        color: red;
    }
    span.MALIGNANT_WITH_CALLBACK {
        color: orange;
    }

    div {
        height: 100%;

        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .calendar-container {
        display: flex;
        flex-direction: row;
        justify-content: center;

        width: 100%;

        margin-top: 1rem;
    }
</style>
