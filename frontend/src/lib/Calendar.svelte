<script lang="ts">
	import { createEventDispatcher } from 'svelte';
    import { Classification } from '../types';

	const headers = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
	const days = [];
	export let severity: keyof typeof Classification;

	const dispatch = createEventDispatcher();

    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth();

    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    let monthAbbrev = monthNames[month].slice(0,3);
    let nextMonthAbbrev = monthNames[(month+1)%12].slice(0,3);
    //	find the last Monday of the previous month
    let firstDay = new Date(year, month, 1).getDay();
    let daysInThisMonth = new Date(year, month+1, 0).getDate();
    let daysInLastMonth = new Date(year, month, 0).getDate();
    let prevMonth = month == 0 ? 11 : month-1;

    //	show the days before the start of this month (disabled) - always less than 7
    for (let i = daysInLastMonth - firstDay; i < daysInLastMonth; i++) {
        let d = new Date(prevMonth == 11 ? year-1 : year, prevMonth, i+1);
        days.push({ name: '' + (i+1), enabled: false, date: d });
    }
    //	show the days in this month (enabled) - always 28 - 31
    for (let i=0; i < daysInThisMonth; i++) {
        let d = new Date(year, month, i+1);
        if (i==0) days.push({ name: monthAbbrev + ' ' + (i+1), enabled: true, date: d });
        else days.push({ name: '' + (i+1), enabled: true, date: d });
    }
    //	show any days to fill up the last row (disabled) - always less than 7
    for (let i=0; days.length % 7; i++) {
        let d = new Date((month == 11 ? year+1 : year), (month+1) % 12, i+1);
        if (i==0) days.push({ name: nextMonthAbbrev + ' ' + (i+1), enabled: false, date: d });
        else days.push({ name: '' + (i+1), enabled: false, date: d });
    }
</script>

<div class="calendar">
	{#each headers as header}
	    <span class="day-name">{header}</span>
	{/each}

	{#each days as day}
		{#if day.enabled}
			<span class="day">{day.name}</span>
		{:else}
			<span class="day day-disabled">{day.name}</span>
		{/if}
	{/each}

    {#if severity === Classification.MALIGNANT || severity === Classification.MALIGNANT_WITH_CALLBACK}
        {#each [1, 2, 5, 8, 9, 14, 15, 16, 20, 29, 30] as opening}
            <section
                class="task {[1,2,9,15,29].includes(opening) ? 'task--warning' : 'task--danger'}"
                style="grid-area: {Math.floor((opening + 4) / 7) + 2} / {(opening + 4) % 7} / auto / span 1;"
            >
                {['Dr. Abernaby', 'Dr. Guillemotte', 'Dr. Bravado'][Math.round(Math.random() * 2)]}
            </section>
        {/each}
    {:else}
        {#each [1, 2, 9, 15, 29] as opening}
            <section
                class="task task--warning"
                style="grid-area: {Math.floor((opening + 4) / 7) + 2} / {(opening + 4) % 7} / auto / span 1;"
            >
                {['Dr. Abernaby', 'Dr. Guillemotte', 'Dr. Bravado'][Math.round(Math.random() * 2)]}
            </section>
        {/each}
    {/if}
</div>

<style>
.calendar {
  display: grid;
  width: 100%;
  grid-template-columns: repeat(7, minmax(120px, 1fr));
  grid-template-rows: 50px;
  grid-auto-rows: 120px;
  overflow: auto;
}
.day {
  border-bottom: 1px solid rgba(166, 168, 179, 0.12);
  border-right: 1px solid rgba(166, 168, 179, 0.12);
  text-align: right;
  padding: 14px 20px;
  letter-spacing: 1px;
  font-size: 14px;
  box-sizing: border-box;
  color: #98a0a6;
  position: relative;
  z-index: 1;
}
.day:nth-of-type(7n + 7) {
  border-right: 0;
}
.day:nth-of-type(n + 1):nth-of-type(-n + 7) {
  grid-row: 1;
}
.day:nth-of-type(n + 8):nth-of-type(-n + 14) {
  grid-row: 2;
}
.day:nth-of-type(n + 15):nth-of-type(-n + 21) {
  grid-row: 3;
}
.day:nth-of-type(n + 22):nth-of-type(-n + 28) {
  grid-row: 4;
}
.day:nth-of-type(n + 29):nth-of-type(-n + 35) {
  grid-row: 5;
}
.day:nth-of-type(n + 36):nth-of-type(-n + 42) {
  grid-row: 6;
}
.day:nth-of-type(7n + 1) {
  grid-column: 1/1;
}
.day:nth-of-type(7n + 2) {
  grid-column: 2/2;
}
.day:nth-of-type(7n + 3) {
  grid-column: 3/3;
}
.day:nth-of-type(7n + 4) {
  grid-column: 4/4;
}
.day:nth-of-type(7n + 5) {
  grid-column: 5/5;
}
.day:nth-of-type(7n + 6) {
  grid-column: 6/6;
}
.day:nth-of-type(7n + 7) {
  grid-column: 7/7;
}
.day-name {
  font-size: 12px;
  text-transform: uppercase;
  color: #e9a1a7;
  text-align: center;
  border-bottom: 1px solid rgba(166, 168, 179, 0.12);
  line-height: 50px;
  font-weight: 500;
}
.day-disabled {
  color: rgba(152, 160, 166, 0.5);
  background-color: #ffffff;
  background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23fdf9ff' fill-opacity='1' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
  cursor: not-allowed;
}

.task {
  border-left-width: 3px;
  padding: 8px 12px;
  margin: 10px;
  border-left-style: solid;
  font-size: 14px;
  position: relative;
  align-self: end;
	z-index:2;
	border-radius: 15px;
}
.task--warning {
  border-left-color: #fdb44d;
  background: #fef0db;
  color: #fc9b10;
  margin-top: -5px;
}
.task--danger {
  border-left-color: #fa607e;
  grid-column: 2 / span 3;
  grid-row: 3;
  margin-top: 15px;
  background: rgba(253, 197, 208, 0.7);
  color: #f8254e;
}
.task--info {
  border-left-color: #4786ff;
  margin-top: 15px;
  background: rgba(218, 231, 255, 0.7);
  color: #0a5eff;
}
.task--primary {
  background: #4786ff;
  border: 0;
  border-radius: 14px;
  color: #f00;
  box-shadow: 0 10px 14px rgba(71, 134, 255, 0.4);
}
.task-detail {
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  background: #efe;
  border: 1px solid rgba(166, 168, 179, 0.2);
  color: #100;
  padding: 20px;
  box-sizing: border-box;
  border-radius: 14px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
  z-index: 2;
}
.task-detail:after, .task-detail:before {
  bottom: 100%;
  left: 30%;
  border: solid transparent;
  content: " ";
  height: 0;
  width: 0;
  position: absolute;
  pointer-events: none;
}
.task-detail:before {
  border-bottom-color: rgba(166, 168, 179, 0.2);
  border-width: 8px;
  margin-left: -8px;
}
.task-detail:after {
  border-bottom-color: #fff;
  border-width: 6px;
  margin-left: -6px;
}
.task-detail h2 {
  font-size: 15px;
  margin: 0;
  color: #91565d;
}
.task-detail p {
  margin-top: 4px;
  font-size: 12px;
  margin-bottom: 0;
  font-weight: 500;
  color: rgba(81, 86, 93, 0.7);
}

</style>
