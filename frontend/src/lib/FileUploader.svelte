<script lang="ts">
    import { Jumper } from 'svelte-loading-spinners';

    import type { Result } from '../types';
    import { Classification } from '../types';

    const endpoint_url = `http://${import.meta.env.VITE_BACKEND_HOST}:${import.meta.env.VITE_BACKEND_PORT}/server/images`;

    let input;
    let isLoading = false;
    export let done: boolean;
    export let result: Result;

    let dragActive: boolean = false;
    const handleDrag = function(e: DragEvent) {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            dragActive = true;
        } else {
            dragActive = false;
        }
    };
    const handleDrop = function(e: DragEvent) {
        e.preventDefault();
        e.stopPropagation();

        isLoading = true;
        dragActive = false;

        if (e.dataTransfer.files?.[0]) {
            const reader = new FileReader();
            reader.readAsDataURL(e.dataTransfer.files[0]);

            reader.onload = () => {
                fetch(endpoint_url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: reader.result }),
                })
                    .then(response => response.json())
                    .then(response => {
                        if (response.status !== 200) throw Error("Request failed");

                        console.log(response);
                        isLoading = false;
                        done = true;
                        result = { classification: response.classification };
                    })
                    .catch(error => {
                        console.error(error);
                        isLoading = false;
                    });
            };
        }
    };

    const handleChange = function(e: Event) {
        e.preventDefault();

        isLoading = true;

        if ((<HTMLInputElement>e.target).files?.[0]) {
            const reader = new FileReader();
            reader.readAsDataURL((<HTMLInputElement>e.target).files[0]);

            reader.onload = () => {
                fetch(endpoint_url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: reader.result }),
                })
                    .then(response => response.json())
                    .then(response => {
                        if (response.status !== 200) throw Error("Request failed");

                        console.log(response);
                        isLoading = false;
                        done = true;
                        result = { classification: response.classification };
                    })
                    .catch(error => {
                        console.error(error);
                        isLoading = false;
                    });
            };
        }
    };

    const onButtonClick = function() {
        input.click();
        done = true;
    }
</script>

{#if isLoading}
<div id="loading">
    <Jumper size="60" color="#ff3e00" unit="px" duration="1s" />
</div>
{:else}
<form id="form-file-uploader" on:dragenter={handleDrag} on:submit={(e) => e.preventDefault()}>
    <input
        bind:this={input}
        type="file"
        id="input-file-uploader"
        on:change={handleChange}
    />
    <label id="label-file-uploader" class={dragActive ? 'drag-active' : ''} for="input-file-uploader">
        <div>
            <p>Drag and drop your file here or</p>
            <button class="upload-button" on:click={onButtonClick}>select a file from your computer</button>
        </div>
    </label>
    {#if dragActive}
        <div
            id="drag-file-element"
            on:dragenter={handleDrag}
            on:dragleave={handleDrag}
            on:dragover={handleDrag}
            on:drop={handleDrop}
        ></div>
    {/if}
</form>
{/if}

<style>
    #loading {
        display: flex;
        justify-content: center;
        align-items: center;
        align-content: center;

        height: 100%;
    }

    #form-file-uploader {
        height: 100%;
        width: 100%;
        max-width: 100%;
        text-align: center;
        position: relative;
    }

    #input-file-uploader {
        display: none;
    }

    #label-file-uploader {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        border-width: 2px;
        border-radius: 1rem;
        border-style: dashed;
        border-color: #cbd5e1;
        background-color: #f8fafc;
        color: #000;
    }
    #label-file-uploader.drag-active {
        background-color: #fff;
    }

    .upload-button {
        cursor: pointer;
        padding: 0.25rem;
        font-size: 1rem;
        border: none;
        font-family: 'Oswald', sans-serif;
        background-color: transparent;
        color: #000;
    }
    .upload-button:hover {
        text-decoration-line: underline;
    }

    #drag-file-element {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 1rem;
        top: 0px;
        right: 0px;
        bottom: 0px;
        left: 0px;
    }
</style>
